"""The parallel branch's rule: which node goes on it and what it waits for (stepgraph.branch_choice),
held without a GPU on the shapes the step has."""
import unittest

from . import stepgraph


class BranchChoiceTest(unittest.TestCase):
    def test_island_after_dense_overlaps_it(self):
        # the attention's key/value projection beside the query projection: both read the node before
        on_par, wait_last, _ = stepgraph.branch_choice("dense", {"a"}, "q", "dense", False, enabled=True)
        self.assertTrue(on_par)
        self.assertFalse(wait_last)

    def test_reader_of_main_last_stays_on_main(self):
        on_par, _, _ = stepgraph.branch_choice("dense", {"q"}, "q", "dense", False, enabled=True)
        self.assertFalse(on_par)

    def test_island_after_host_node_waits_for_it(self):
        # the shared expert's front beside the routed experts: on the branch, but after the host node
        on_par, wait_last, _ = stepgraph.branch_choice("dense", {"front"}, "experts", "experts", False, enabled=True)
        self.assertTrue(on_par)
        self.assertTrue(wait_last)

    def test_host_node_waits_for_the_branch(self):
        on_par, _, host_waits = stepgraph.branch_choice("experts", {"front"}, "front", "dense", False, enabled=True)
        self.assertFalse(on_par)
        self.assertTrue(host_waits)

    def test_a_wave_from_another_gpu_keeps_a_node_on_main(self):
        on_par, _, _ = stepgraph.branch_choice("dense", {"a"}, "q", "dense", True, enabled=True)
        self.assertFalse(on_par)

    def test_disabled(self):
        on_par, _, _ = stepgraph.branch_choice("dense", {"a"}, "q", "dense", False, enabled=False)
        self.assertFalse(on_par)


if __name__ == "__main__":
    unittest.main()
