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

"""Process-per-rank execution on the CPU (#539, slice 2): the coordinator spawns one executor
per rank on the selftest's tiny-deepseek-v41 exported with --tp 2 (the tensor-parallel shape
the selftest runs on the CPU), drives the prefill and 24 greedy steps through the messages,
and the tokens must be the one-process engine's (the same two ranks on two threads), bit for
bit. The single-rank engine is not that reference: on this tiny random model it flips a token
whose top two logits sit 0.0015 apart against the two-rank split (the selftest's tensor parallel
check excuses exactly that, a flip inside the measured tie band). Also: a dead executor fails
the coordinator quickly, and the exchange's tensor packing round-trips every dtype. No GPU; the
fork is needed for the fixture."""

import os
import statistics
import tempfile
import unittest

import torch

from . import world

PROMPTS = [[(11 * i + 7 * j + 5) % 30000 + 100 for j in range(n)] for i, n in enumerate((9, 14))]
ENV = {"VXPU_DENSE_TP": "2", "VXPU_DATAFLOW": "1", "VXPU_CHECK_GOLDEN": "0"}


def _fork_available():
    try:
        from ..deepseek_v41 import _fork
        _fork()
        return True
    except ImportError:
        return False


class PackTest(unittest.TestCase):
    def test_tensor_round_trip(self):
        for x in (torch.randn(3, 8), torch.randn(2, 5).bfloat16(), torch.arange(6).reshape(2, 3), torch.empty(0, 4),
                  torch.tensor(2.5), torch.randn(4).half(), torch.tensor([True, False])):
            y = world.unpack_tensor(world.pack_tensor(x))
            self.assertEqual(y.dtype, x.dtype)
            self.assertEqual(tuple(y.shape), tuple(x.shape))
            self.assertTrue(torch.equal(x, y))


@unittest.skipUnless(_fork_available(), "the DeepSeek-V4.1 fork is not checked out (vxpu/third_party/deepseek_v41)")
class CoordinatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ..fixtures import tiny_deepseek_v41
        from ..export import export_artifact
        cls.tmp = tempfile.TemporaryDirectory(prefix="vxpu-ppr-")
        model = tiny_deepseek_v41.write(os.path.join(cls.tmp.name, "tiny-deepseek-v41"))
        cls.tp2 = os.path.join(cls.tmp.name, "artifact-tp2")
        export_artifact(model, cls.tp2, max_cache_len=256, batch=1, tp=2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _with_env(self, env, fn):
        previous = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            return fn()
        finally:
            for k, v in previous.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_two_processes_decode_the_one_process_engine_tokens(self):
        from ..engine import Engine, SamplingParams
        from .coordinator import Coordinator
        greedy = SamplingParams(do_sample=False)

        def threads():
            engine = Engine(self.tp2)
            self.assertEqual(len(engine._tp_graphs), 2, "the thread world still builds every rank")
            return engine._generate_rows_tp(PROMPTS, 24, greedy)
        reference = self._with_env(ENV, threads)

        with Coordinator(self.tp2, 2, env=ENV, log_dir=self.tmp.name) as c:
            got = c.generate(PROMPTS, 24)
            steps = c.steps
        self.assertEqual(got["token_ids"], reference["token_ids"], "two processes against two threads")
        # every prompt token fed one at a time, then max_new_tokens steps a group (the engine's loop count)
        self.assertEqual(steps, sum(map(len, PROMPTS)) + len(PROMPTS) * 24)
        print(f"\nppr: 2 executors loaded in {c.load_seconds:.1f} s; {steps} steps; {sum(map(len, got['token_ids']))} tokens "
              f"identical to the one-process engine's; step {statistics.median(got['step_ms']):.1f} ms median against "
              f"{statistics.median(reference['step_ms']):.1f} on two threads")

    def test_a_dead_executor_fails_the_coordinator(self):
        from .coordinator import Coordinator
        with self.assertRaises(RuntimeError) as caught:
            Coordinator(self.tp2, 2, env=dict(ENV, VXPU_DEEPSEEK_V41="/nonexistent"), log_dir=self.tmp.name)
        self.assertIn("exited", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
