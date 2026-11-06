# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The sweep-and-write tool without a card: the racer's bookkeeping with a fake timer (the
candidates as the Go planner enumerates them, the engine's own blocks first, failures counted,
the stages pass on the winner), the table writer round-tripping through plan.load_table over an
existing table, and the command's dry run writing nothing."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from . import plan, tune

QUANT = "gemv_rows rows=128 n=5120 k=8192 w=fp8 epi=quant native=1"
SPLIT = "gemv_rows rows=1 n=1280 k=5120 w=fp8 native=1 split=1"
DUAL = "dual_gemv_swiglu rows=9 n=2304 k=5120 w=fp8 native=1"


def _fake_timer(key, blocks):
    """Deterministic: BN 32 fastest, BK 256 next, 8 warps a little slower; BK 512 on the quant class fails."""
    if "epi=quant" in key and blocks["BK"] == 512:
        raise RuntimeError("out of resource: shared memory")
    return 10.0 + abs(blocks["BN"] - 32) / 8 + abs(blocks["BK"] - 256) / 128 + (0.5 if blocks["warps"] == 8 else 0) \
        + (0.3 if blocks["stages"] != 3 else 0) + (0.2 if blocks["KSPLIT"] != 1 else 0)


class CandidatesTest(unittest.TestCase):
    def test_go_rules(self):
        from .measure import parse_class
        quant = tune.candidates(parse_class(QUANT))
        self.assertTrue(all(b["BN"] in (32, 64) for b in quant), "the quantizing epilogue holds whole blocks of 32")
        self.assertTrue(all(b["KSPLIT"] == 1 for b in quant), "no K split where the caller takes no partial sums")
        self.assertEqual({b["warps"] for b in quant}, {4, 8}, "a 40 MB weight: 4 and 8 warps")
        split = tune.candidates(parse_class(SPLIT), sms=148)
        self.assertEqual({b["warps"] for b in split}, {2, 4, 8}, "a weight under 16 MB tries 2 warps too")
        self.assertTrue({b["KSPLIT"] for b in split} >= {1, 2, 4})
        self.assertTrue(all((1280 // b["BN"]) * (b["KSPLIT"] // 2) < 2 * 148 for b in split if b["KSPLIT"] > 1),
                        "a split past twice the card's SMs is not tried")
        own = {"BN": 64, "BK": 512, "KSPLIT": 1, "warps": 4, "stages": 3}
        first = tune.candidates(parse_class(QUANT), ran=own)[0]
        self.assertEqual(tune.block_key(first), tune.block_key(own), "the engine's own blocks are raced first")
        self.assertEqual(len({tune.block_key(b) for b in quant}), len(quant), "no duplicates")


class RaceTest(unittest.TestCase):
    def test_bookkeeping_with_a_fake_timer(self):
        ran = {QUANT: {"BN": 64, "BK": 128, "KSPLIT": 1, "warps": 4, "stages": 3}}
        raced = tune.race([QUANT, SPLIT], _fake_timer, ran=ran, sms=148, floors={QUANT: 8.0})
        q = raced[QUANT]
        self.assertEqual((q["best"]["BN"], q["best"]["BK"], q["best"]["warps"], q["best"]["stages"]), (32, 256, 4, 3))
        self.assertAlmostEqual(q["us"], 10.0)
        self.assertEqual(q["ran"], ran[QUANT])
        self.assertAlmostEqual(q["ran_us"], _fake_timer(QUANT, ran[QUANT]))
        self.assertEqual(q["failed"], sum(1 for b, us, err in q["results"] if b["BK"] == 512), "BK 512 failed on this class")
        self.assertTrue(all(err for b, us, err in q["results"] if b["BK"] == 512))
        stages = [b["stages"] for b, us, _ in q["results"] if b["BN"] == 32 and b["BK"] == 256 and b["warps"] == 4]
        self.assertEqual(sorted(stages), [2, 3, 4], "the depth pass ran on the winner")
        s = raced[SPLIT]
        self.assertEqual(s["ran"], tune.default_blocks(__import__("vxpu.measure", fromlist=["x"]).parse_class(SPLIT)))
        self.assertEqual(s["best"]["KSPLIT"], 1)
        lines = tune.report(raced, {QUANT: 230})
        self.assertIn("(230 a step)", lines[0])
        self.assertIn("floor 8.0 us (80% of bandwidth)", lines[0])
        self.assertIn("current BK=128 BN=64 KSPLIT=1 stages=3 warps=4", lines[0])
        self.assertIn("best BK=256 BN=32 KSPLIT=1 stages=3 warps=4 10.0 us", lines[0])

    def test_no_candidate_running_keeps_the_engine_blocks(self):
        def broken(key, blocks):
            raise RuntimeError("no card")
        raced = tune.race([DUAL], broken)
        self.assertIsNone(raced[DUAL]["best"])
        self.assertEqual(tune.entries(raced, lambda k: 1), [])
        self.assertIn("no candidate ran", tune.report(raced)[0])


class TableTest(unittest.TestCase):
    def test_round_trip_over_an_existing_table(self):
        from .measure import class_bytes
        existing = {"format": plan.TABLE_FORMAT, "gpu": "old card", "knobs": {"GEMV_ROWS_MAX": 64},
                    "blocks": [{"shapes": QUANT, "blocks": {"BN": 64, "BK": 128, "KSPLIT": 1, "warps": 4, "stages": 3}},
                               {"shapes": DUAL, "blocks": {"BN": 32, "BK": 128, "KSPLIT": 8, "warps": 2, "stages": 4}}]}
        raced = tune.race([QUANT, SPLIT], _fake_timer, sms=148, floors={QUANT: 8.0, SPLIT: 4.5})
        new = tune.entries(raced, class_bytes, bands={QUANT: [96, 128]})
        self.assertEqual(sorted(new[0]), ["blocks", "bytes", "efficiency", "floor_us", "rows_band", "shapes", "us", "why"])
        self.assertEqual(sorted(new[0]["blocks"]), ["BK", "BN", "KSPLIT", "stages", "warps"])
        merged = tune.merge_table(existing, new, gpu="NVIDIA RTX PRO 6000", explain=tune.report(raced))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "table.json")
            tune.write_table(merged, path)
            loaded = plan.load_table(path)
            with open(path) as f:
                raw = json.load(f)
        self.assertEqual(raw["format"], plan.TABLE_FORMAT)
        self.assertEqual(raw["knobs"], {"GEMV_ROWS_MAX": 64}, "the other sections are kept")
        self.assertEqual([b["shapes"] for b in raw["blocks"]], sorted([DUAL, QUANT, SPLIT]), "sorted by class, DUAL kept")
        self.assertEqual(loaded["blocks_index"][QUANT], {"BN": 32, "BK": 256, "KSPLIT": 1, "warps": 4, "stages": 3})
        self.assertEqual(loaded["blocks_index"][DUAL]["KSPLIT"], 8)
        self.assertEqual(loaded["blocks_bands"], [("gemv_rows n=5120 k=8192 w=fp8 epi=quant native=1", 96, 128,
                                                   {"BN": 32, "BK": 256, "KSPLIT": 1, "warps": 4, "stages": 3})])
        self.assertTrue(raw["explain"][0].startswith(QUANT))
        plan.apply_blocks(loaded["blocks_index"], loaded["blocks_bands"])  # what the engine does with it
        try:
            from . import triton_ops
            banded = "gemv_rows rows=112 n=5120 k=8192 w=fp8 epi=quant native=1"
            self.assertEqual(triton_ops.gemv_blocks(banded, 16, 64, 128)[1:3], (32, 256), "112 rows take the band's tiles")
        finally:
            plan.apply_blocks({}, [])


class CommandTest(unittest.TestCase):
    def test_dry_run_prints_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            shapes = os.path.join(tmp, "shapes.json")
            with open(shapes, "w") as f:
                json.dump({"classes": [QUANT], "bands": {QUANT: [96, 128]}}, f)
            out_path = os.path.join(tmp, "table.json")
            stdout, stderr = io.StringIO(), io.StringIO()
            previous = os.environ.get("VXPU_TUNE_SWEEP")
            os.environ["VXPU_TUNE_SWEEP"] = shapes  # the knob names the shapes file
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = tune.main(["--dry-run", "--out", out_path, "--gbps", "1460"], timer=_fake_timer, gpu="fake card", sms=148)
            finally:
                if previous is None:
                    os.environ.pop("VXPU_TUNE_SWEEP", None)
                else:
                    os.environ["VXPU_TUNE_SWEEP"] = previous
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(out_path), "a dry run writes nothing")
            text = stdout.getvalue()
            self.assertIn("dry run: 1 entries not written", text)
            self.assertIn("best BK=256 BN=32", text)
            self.assertIn('"rows_band"', text)
            self.assertIn("[tune] " + QUANT, stderr.getvalue())
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = tune.main([shapes, "--out", out_path, "--gbps", "1460"], timer=_fake_timer, gpu="fake card", sms=148)
            self.assertEqual(rc, 0)
            self.assertEqual(plan.load_table(out_path)["blocks_index"][QUANT]["BN"], 32)

    def test_a_kernel_record_is_a_shapes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = os.path.join(tmp, "record.json")
            with open(record, "w") as f:
                json.dump({"format": plan.RECORD_FORMAT, "stats": {"gemv_classes": {
                    QUANT: {"count": 500, "sites": 230, "blocks": {"BM": 128, "BN": 64, "BK": 128, "KSPLIT": 1, "warps": 4, "stages": 3}}}}}, f)
            classes, ran, bands, a_step = tune.read_shapes(record)
            self.assertEqual(classes, [QUANT])
            self.assertEqual(ran[QUANT]["BN"], 64)
            self.assertEqual(a_step, {QUANT: 230})


if __name__ == "__main__":
    unittest.main()
