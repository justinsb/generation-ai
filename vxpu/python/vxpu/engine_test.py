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

"""The engine's expert statistics over the step graphs of every batch in flight (hero run 12)."""

import unittest

from . import engine


class _Graph:
    def __init__(self, counts):
        self._counts = counts

    def expert_stats(self):
        return self._counts


class ExpertStats(unittest.TestCase):
    def test_sums_every_batch_graph_and_skips_graphs_that_do_not_count(self):
        own = _Graph({0: 6, 1: 6})
        batch1 = _Graph({0: 30, 1: 36})
        drafter = _Graph(None)  # the drafter's stages have no experts
        self.assertEqual(engine.sum_expert_stats([own, batch1, drafter, None]), {0: 36, 1: 42})
        self.assertIsNone(engine.sum_expert_stats([drafter, None]))

    def test_the_engine_reads_its_own_graph_and_the_pipeline_graphs(self):
        e = engine.Engine.__new__(engine.Engine)
        e._decode_split = _Graph({0: 6})
        e._pipeline_graphs = [(None, _Graph({0: 30}), _Graph(None)), (None, _Graph({0: 37}), None)]
        self.assertEqual(e.expert_stats(), {0: 73})
        del e._pipeline_graphs
        self.assertEqual(e.expert_stats(), {0: 6})


class _Plan(dict):
    """A plan for a Checks object: the switches, and the record and note calls as no-ops."""

    def stat(self, name, value):
        self[("stat", name)] = value

    def note(self, text):
        self.setdefault(("notes",), []).append(text)


class _Recording:
    """A checks object that records what the tensor-parallel loop asks of it."""

    def __init__(self, due_every):
        self.calls = []
        self.due_every = due_every
        self.duplicate_rows = 1

    def tp_shadow(self, n, rows):
        self.calls.append(("tp_shadow", n, rows))
        return 0 < n < rows

    def tp_compare(self, chosen, original, shadow, step):
        self.calls.append(("tp_compare", step))

    def reference_due(self, steps):
        self.calls.append(("reference_due", steps))
        return steps % self.due_every == 0

    def reference_tp(self, ids, positions, active, logits, steps, live):
        self.calls.append(("reference_tp", steps, tuple(active)))
        return True


def _tp_engine(rows, checks, step_fn):
    """An engine with only what _generate_rows_tp reads: the group size, a tensor-parallel step
    (step_fn(ids, positions) -> logits [rows, vocab]), one rank's empty state, and the checks."""
    import contextlib
    import threading
    e = engine.Engine.__new__(engine.Engine)
    e._draft_run, e.step_tokens = None, 1
    e.generation_config = {}
    e.binding = {"batch": rows}
    e._lock = threading.Lock()
    e.external_scope = contextlib.nullcontext
    e._tp_states = [{}]
    e._resident = None
    e.eos_ids = {1}
    e._decode_run = None
    e._tp_self_fed_ready = lambda: False
    e._tp_step = step_fn
    e.checks = checks
    return e


def _next_token_logits(rows, vocab, shift_row=None):
    """A step whose every row chooses its fed token + 1 (so a copy of a row chooses what the row
    chooses), except shift_row, which chooses one more: the duplicate-row invariant broken."""
    import torch

    def step(ids, positions):
        out = torch.zeros(rows, vocab)
        for r, (tok,) in enumerate(ids):
            out[r, (tok + 1 + (1 if r == shift_row else 0)) % vocab] = 1.0
        return out
    return step


class TensorParallelLoopChecks(unittest.TestCase):
    """The standing checks run on the tensor-parallel loop (hero run 15: they had been wired into
    the other row loops only, and the TP step's measurements ran for days without them)."""

    def _checks(self, **switches):
        from . import checks
        plan = _Plan({"check_golden": False, "check_duplicate_rows": 1, "check_reference_every": 0,
                      "check_reference_budget": 0.02, "check_on_mismatch": "warn"})
        plan.update(switches)
        e = engine.Engine.__new__(engine.Engine)
        e.plan = plan
        return checks.Checks(e), plan

    def test_a_free_row_carries_a_copy_of_row_0_and_the_copy_is_compared_every_step(self):
        c, plan = self._checks()
        e = _tp_engine(3, c, _next_token_logits(3, 16))
        e.plan, c.engine = plan, e
        out = e._generate_rows_tp([[5, 6, 7], [8, 9]], 3, engine.SamplingParams(do_sample=False))
        self.assertEqual(out["token_ids"], [[8, 9, 10], [10, 11, 12]])
        self.assertEqual(out["padded_rows"], 1)
        self.assertEqual(c.results["duplicate_rows"]["padded"], 1)
        self.assertEqual(c.results["duplicate_rows"]["compared"], 4)  # after the prefill and after each of the 3 steps
        self.assertEqual(c.results["duplicate_rows"]["mismatches"], 0)

    def test_a_copy_that_chooses_differently_is_a_mismatch_the_plan_warns_about(self):
        c, plan = self._checks()
        e = _tp_engine(3, c, _next_token_logits(3, 16, shift_row=2))  # row 2 is the copy of row 0
        e.plan, c.engine = plan, e
        out = e._generate_rows_tp([[5, 6, 7], [8, 9]], 2, engine.SamplingParams(do_sample=False))
        self.assertEqual(out["token_ids"], [[8, 9], [10, 11]])  # the copy never reaches the results
        self.assertEqual(c.results["duplicate_rows"]["mismatches"], 3)
        self.assertTrue(any("duplicate row" in n for n in plan.get(("notes",), [])))

    def test_no_free_row_means_no_copy(self):
        c, plan = self._checks()
        e = _tp_engine(2, c, _next_token_logits(2, 16))
        e.plan, c.engine = plan, e
        out = e._generate_rows_tp([[5, 6, 7], [8, 9]], 2, engine.SamplingParams(do_sample=False))
        self.assertEqual(out["padded_rows"], 0)
        self.assertEqual(c.results["duplicate_rows"]["padded"], 0)

    def test_the_loop_asks_for_the_reference_check_every_step_and_runs_it_when_due(self):
        rec = _Recording(due_every=2)
        e = _tp_engine(3, rec, _next_token_logits(3, 16))
        e._generate_rows_tp([[5, 6, 7], [8, 9]], 4, engine.SamplingParams(do_sample=False))
        asked = [c[1] for c in rec.calls if c[0] == "reference_due"]
        self.assertEqual(asked, [1, 2, 3, 4])
        ran = [c for c in rec.calls if c[0] == "reference_tp"]
        self.assertEqual([c[1] for c in ran], [2, 4])
        self.assertEqual(ran[0][2], (0, 1))  # both rows active
        self.assertEqual([c for c in rec.calls if c[0] == "tp_shadow"], [("tp_shadow", 2, 3)])
        self.assertEqual(len([c for c in rec.calls if c[0] == "tp_compare"]), 5)


if __name__ == "__main__":
    unittest.main()


class SpecVerifyRows(unittest.TestCase):
    """The verify rows the fixed, half and flat shapes build on the device, on CPU tensors: a fixed draft under
    the drafter's block issues exactly that many drafts a row into the program's width (hero run 15: the mask
    had the policy's width and the drafts the block's, and the step failed at draft 5 on a block-7 drafter)."""

    def test_fixed_depth_under_the_block_fills_the_program_width(self):
        import torch
        R, block, vrows = 3, 7, 8
        first = torch.tensor([[100], [200], [300]])
        dout = torch.cat([torch.arange(R * block).view(R, block) + 1, torch.zeros(R, block, dtype=torch.long)], 1)
        for counts in ([5, 5, 5], [7, 7, 7], [5, 0, 7]):
            ids, mask = engine._spec_verify_rows(first, dout, torch.tensor(counts), vrows, block)
            self.assertEqual(tuple(ids.shape), (R, vrows))
            self.assertEqual(tuple(mask.shape), (R, vrows - 1))
            self.assertEqual(mask.sum(1).tolist(), counts)
            for i, c in enumerate(counts):
                self.assertEqual(ids[i, 0].item(), first[i].item())
                self.assertEqual(ids[i, 1:c + 1].tolist(), dout[i, :c].tolist())  # the issued drafts
                self.assertEqual(ids[i, c + 1:].tolist(), [first[i].item()] * (vrows - 1 - c))  # the rest, the token
        ids, mask = engine._spec_verify_rows(first, dout, torch.tensor([3, 3, 3]), 10, block)  # a program wider than the block
        self.assertEqual(tuple(ids.shape), (R, 10))
        self.assertEqual(mask.sum(1).tolist(), [3, 3, 3])


def _fork_available():
    try:
        from .deepseek_v41 import _fork
        _fork()
        return True
    except ImportError:
        return False


@unittest.skipUnless(_fork_available(), "the DeepSeek-V4.1 fork is not checked out (vxpu/third_party/deepseek_v41)")
class FixedDepthUnderTheBlock(unittest.TestCase):
    """tiny-deepseek-v41 exported with --draft --draft-block 7 --step-tokens 8 at tp 2 (the selftest's CPU shape):
    the fixed policy at draft 5 issues 5 a live row a step, at draft 7 issues 7, and both decode the undrafted
    run's tokens (VXPU_NO_DRAFTS=1 through the same programs), the exactness the selftest checks for the drafted
    shapes. On the CPU the drafted step runs its host path; the device path's rows are SpecVerifyRows."""

    def test_fixed_5_and_7_match_the_undrafted_tokens(self):
        import json
        import os
        import tempfile
        from . import physplan
        from .engine import Engine, SamplingParams
        from .export import export_artifact
        from .fixtures import tiny_deepseek_v41
        prompts = [[(11 * i + 7 * j + 5) % 30000 + 100 for j in range(n)] for i, n in enumerate((9, 14))]
        greedy = SamplingParams(do_sample=False)
        keys = ("VXPU_DENSE_TP", "VXPU_DATAFLOW", "VXPU_CHECK_GOLDEN", "VXPU_PLAN", "VXPU_NO_DRAFTS")
        previous = {k: os.environ.get(k) for k in keys}
        with tempfile.TemporaryDirectory(prefix="vxpu-fixed-") as tmp:
            model = tiny_deepseek_v41.write(os.path.join(tmp, "tiny-deepseek-v41"))
            artifact = os.path.join(tmp, "artifact-tp2-b7s8")
            export_artifact(model, artifact, max_cache_len=256, batch=1, step_tokens=8, draft=True, draft_block=7, tp=2)
            os.environ.update({"VXPU_DENSE_TP": "2", "VXPU_DATAFLOW": "1", "VXPU_CHECK_GOLDEN": "0"})

            def run(draft, no_drafts=False):
                plan = os.path.join(tmp, f"plan{draft}{no_drafts}.json")
                with open(plan, "w") as f:
                    json.dump({"format": "vxpu-plan/1", "speculation": {"draft": draft, "policy": "fixed"}}, f)
                os.environ["VXPU_PLAN"] = plan
                if no_drafts:
                    os.environ["VXPU_NO_DRAFTS"] = "1"
                else:
                    os.environ.pop("VXPU_NO_DRAFTS", None)
                physplan._loaded = None
                try:
                    return Engine(artifact)._generate_rows_tp(prompts, 16, greedy)
                finally:
                    physplan._loaded = None
            try:
                plain = run(7, no_drafts=True)
                five, seven = run(5), run(7)
            finally:
                for k, v in previous.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        self.assertEqual(plain["drafted"], 0)
        self.assertEqual(five["token_ids"], plain["token_ids"], "fixed 5 against the undrafted run")
        self.assertEqual(seven["token_ids"], plain["token_ids"], "fixed 7 against the undrafted run")
        self.assertEqual(five["drafted"], 5 * sum(five["rows_active"]), "5 drafts a live row a step")
        self.assertEqual(seven["drafted"], 7 * sum(seven["rows_active"]), "7 drafts a live row a step")
        self.assertGreater(five["accepted"], 0)
        print(f"\nfixed depth under the block: draft 5 issued {five['drafted']} over {sum(five['rows_active'])} row-steps "
              f"({five['accepted']} accepted), draft 7 issued {seven['drafted']}; both the undrafted run's "
              f"{sum(map(len, plain['token_ids']))} tokens")
