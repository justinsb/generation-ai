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

"""segments.split_at_host_ops: the cut around an island op that returns several tensors.

An island op cut into a segment of its own (no fences around it: the lowering did not run, as on
the CPU) returns a tuple. Its segment must return the tensors, unpacked, so that each reaches its
consumers through a top-level getitem the dataflow runner routes through an inbox of its own; a
segment handing its whole tuple to the next one gave the runner a tuple in one inbox, with no shape
to plan and nothing to copy (the CPU selftest's dataflow check died on it).
Run: python -m unittest vxpu.segments_test
"""

import operator
import unittest

import torch

from . import segments


@torch.library.custom_op("vxpu_segments_test::pair", mutates_args=())
def _pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return x * 2, x + 1


@_pair.register_fake
def _pair_fake(x):
    return torch.empty_like(x), torch.empty_like(x)


class Pair(torch.nn.Module):
    def forward(self, x):
        a, b = torch.ops.vxpu_segments_test.pair(x)
        return a + b


class IslandTupleTest(unittest.TestCase):
    def test_unfenced_island_segment_returns_its_tensors(self):
        # exported, as the engine's graphs are: the op appears as its overload (pair.default)
        gm = torch.export.export(Pair(), (torch.zeros(2, 3),), strict=False).module()
        saved = segments.ISLAND_OPS
        segments.ISLAND_OPS = saved + ("vxpu_segments_test.pair.default",)
        try:
            split, host = segments.split_at_host_ops(gm)
        finally:
            segments.ISLAND_OPS = saved
        self.assertEqual(host, [])
        calls = [n for n in split.graph.nodes if n.op == "call_module"]
        self.assertGreaterEqual(len(calls), 2)
        # no segment takes another segment's whole output: every argument is a placeholder, an
        # attribute or a getitem of a segment
        for call in calls:
            for arg in call.args:
                if isinstance(arg, torch.fx.Node):
                    self.assertNotEqual(arg.op, "call_module",
                                        f"{call.name} takes {arg.name}'s whole output")
        # the island's segment holds the op and its getitems, and returns two tensors
        island = next(c for c in calls if any(str(n.target) == "vxpu_segments_test.pair.default"
                                              for n in getattr(split, c.target).graph.nodes))
        sub = getattr(split, island.target)
        self.assertTrue(any(n.op == "call_function" and n.target is operator.getitem for n in sub.graph.nodes))
        out = next(n for n in sub.graph.nodes if n.op == "output")
        self.assertEqual(len(out.args[0]), 2)
        x = torch.arange(6.0).view(2, 3)
        torch.testing.assert_close(split(x)[0] if isinstance(split(x), tuple) else split(x), x * 2 + x + 1)


if __name__ == "__main__":
    unittest.main()
