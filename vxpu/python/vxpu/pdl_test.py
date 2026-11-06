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

"""The chaining-parity knob (VXPU_PDL_ALL=1, hero run 16), GPU-free: the environment reaches the
knob and the plan's record, the knob reaches inductor's config.triton.enable_pdl and moves the
kernel digest (so the packages rebuild), and of our kernels only the ones with a wait inside
take the launch flag (through _fp4_pdl_mode), with an explicit FP4_PDL standing over it."""

import os
import subprocess
import sys
import unittest

from . import engine, lower, plan, triton_ops


class _Plan(dict):
    def __getitem__(self, k):
        return self.get(k, False)


class PdlAllTest(unittest.TestCase):
    def setUp(self):
        self._saved = (triton_ops.PDL_ALL, triton_ops.FP4_PDL, triton_ops._gdc_wait)
        import torch._inductor.config as cfg
        self._cfg = cfg
        self._enable_pdl = getattr(cfg.triton, "enable_pdl", None)

    def tearDown(self):
        triton_ops.PDL_ALL, triton_ops.FP4_PDL, triton_ops._gdc_wait = self._saved
        if self._enable_pdl is not None:
            self._cfg.triton.enable_pdl = self._enable_pdl

    def test_environment_reaches_the_knob_and_the_record(self):
        code = "from vxpu import plan, triton_ops; print(triton_ops.PDL_ALL, plan.effective_knobs()['PDL_ALL'])"
        env = dict(os.environ, VXPU_PDL_ALL="1", PYTHONPATH=os.pathsep.join(sys.path))
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.split(), ["1", "True"])

    def test_knob_reaches_inductor_and_the_digest(self):
        self.assertTrue(hasattr(self._cfg.triton, "enable_pdl"), "torch 2.14's config.triton.enable_pdl")
        triton_ops.PDL_ALL = None
        self.assertFalse(lower._outside_segments("PDL_ALL"), "the knob changes the dense segments' code")
        off = lower.kernel_digest()
        self._cfg.triton.enable_pdl = False
        engine.configure_inductor(_Plan())
        self.assertFalse(self._cfg.triton.enable_pdl)
        plan.apply_knobs({"PDL_ALL": True})  # the table's form
        self.assertEqual(triton_ops.PDL_ALL, "1")
        self.assertTrue(plan.effective_knobs()["PDL_ALL"])
        self.assertNotEqual(lower.kernel_digest(), off, "the packages must rebuild under the knob")
        engine.configure_inductor(_Plan())
        self.assertTrue(self._cfg.triton.enable_pdl)

    def test_only_kernels_with_a_wait_take_the_flag(self):
        triton_ops._gdc_wait = object()  # as on a card with Triton's CUDA backend
        triton_ops.PDL_ALL, triton_ops.FP4_PDL = None, None
        self.assertEqual(triton_ops._fp4_pdl_mode(), 0)
        triton_ops.PDL_ALL = "1"
        self.assertEqual(triton_ops._fp4_pdl_mode(), 1, "the fp4 expert kernels wait inside: they take the flag")
        triton_ops.FP4_PDL = "0"
        self.assertEqual(triton_ops._fp4_pdl_mode(), 0, "an explicit FP4_PDL stands over PDL_ALL")
        triton_ops.FP4_PDL = "2"
        self.assertEqual(triton_ops._fp4_pdl_mode(), 2)
        triton_ops._gdc_wait = None  # no CUDA backend: no dependent launch anywhere
        triton_ops.FP4_PDL = None
        self.assertEqual(triton_ops._fp4_pdl_mode(), 0)


if __name__ == "__main__":
    unittest.main()
