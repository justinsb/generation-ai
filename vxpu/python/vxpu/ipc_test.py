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

"""The handle exchange without a card: N fake ranks on threads share tensors through one
directory with a fake reducer (the tensor pickled), every rank sees every rank's in order, a
stale file of another run is not read, the barrier holds, and the Comm of a process per rank
aliases its peers' boxes through the shared arenas (CPU tensors standing for the mapped ones)."""

import os
import pickle
import tempfile
import threading
import unittest

import torch

from . import allreduce, ipc


def _fake_reduce(t):
    return pickle.dumps(("fake", t.clone()))


def _fake_rebuild(payload, device):
    kind, t = pickle.loads(payload)
    return t


class RetargetTest(unittest.TestCase):
    def test_the_handle_opens_under_this_card(self):
        # rebuild_cuda_tensor's arguments as torch's reducer makes them for a tensor on card 2
        args = (torch.Tensor, (1024,), (1,), 0, torch.UntypedStorage, torch.uint8, 2, b"handle", 1024, 0, False,
                b"ref", 0, b"event", False)
        mine = ipc.retarget(args, torch.device("cuda", 0))
        self.assertEqual(mine[ipc.STORAGE_DEVICE], 0, "opened under this card's context, not the owner's")
        self.assertEqual(mine[:ipc.STORAGE_DEVICE] + mine[ipc.STORAGE_DEVICE + 1:], args[:6] + args[7:], "nothing else moves")
        self.assertEqual(ipc.retarget(args, 3)[ipc.STORAGE_DEVICE], 3)
        self.assertIs(ipc.retarget(args, None), args)

    def test_the_exchange_hands_its_device_to_rebuild(self):
        seen = []

        def rebuild(payload, device):
            seen.append(device)
            return _fake_rebuild(payload, device)
        with tempfile.TemporaryDirectory() as d:
            a = ipc.Exchange(d, 0, 2, "t", reduce=_fake_reduce, rebuild=rebuild, device=torch.device("cuda", 0))
            b = ipc.Exchange(d, 1, 2, "t", reduce=_fake_reduce, rebuild=rebuild, device=torch.device("cuda", 1))
            b.publish("x", _fake_reduce(torch.ones(2)))
            got = a.share("x", torch.zeros(2))
            self.assertEqual(seen, [torch.device("cuda", 0)])
            self.assertTrue(got[1].is_contiguous() and got[1].storage_offset() == 0)


class ExchangeTest(unittest.TestCase):
    def test_every_rank_sees_every_rank_in_order(self):
        n = 4
        with tempfile.TemporaryDirectory() as d:
            got = [None] * n
            errors = []

            def run(r):
                try:
                    ex = ipc.Exchange(d, r, n, "run1", reduce=_fake_reduce, rebuild=_fake_rebuild)
                    mine = torch.full((3,), float(r))
                    shared = ex.share("comm", mine)
                    self.assertIs(shared[r], mine)
                    got[r] = [float(t[0]) for t in shared]
                    ex.barrier("primed")
                    self.assertEqual(len(ex.opened), n - 1)
                    ex.close()
                    self.assertEqual(ex.opened, [])
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
            threads = [threading.Thread(target=run, args=(r,)) for r in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
            self.assertEqual(errors, [])
            self.assertEqual(got, [[0.0, 1.0, 2.0, 3.0]] * n)

    def test_another_run_and_a_missing_peer(self):
        with tempfile.TemporaryDirectory() as d:
            old = ipc.Exchange(d, 1, 2, "old", reduce=_fake_reduce, rebuild=_fake_rebuild)
            old.publish("comm", _fake_reduce(torch.zeros(1)))
            new = ipc.Exchange(d, 0, 2, "new", reduce=_fake_reduce, rebuild=_fake_rebuild)
            with self.assertRaises(TimeoutError):  # rank 1's file is the old run's: not this run's
                new.share("comm", torch.ones(1)) if False else new.gather("comm", timeout=0.2)
            self.assertEqual(ipc.key_name("world", ("step", 3)), "world:('step', 3)")


class WiringTest(unittest.TestCase):
    def test_the_knob(self):
        from . import multiproc
        saved = multiproc.SUMS, os.environ.get("VXPU_MP_WORLD")
        try:
            os.environ.pop("VXPU_MP_WORLD", None)
            multiproc.SUMS = "push"
            self.assertFalse(multiproc.push_sums(), "no process world: no push sums")
            os.environ["VXPU_MP_WORLD"] = "4"
            self.assertTrue(multiproc.push_sums())
            self.assertEqual([d.index for d in multiproc.all_devices()], [0, 1, 2, 3])
            multiproc.SUMS = "nccl"
            self.assertFalse(multiproc.push_sums())
        finally:
            multiproc.SUMS = saved[0]
            if saved[1] is None:
                os.environ.pop("VXPU_MP_WORLD", None)
            else:
                os.environ["VXPU_MP_WORLD"] = saved[1]

    def test_two_worlds_share_the_ring_buffers(self):
        from . import ipc, parallel
        arenas, lock, done = {}, threading.Lock(), threading.Barrier(2)

        def fake_share(name, arena):
            me = threading.current_thread().name
            with lock:
                arenas[(name, me)] = arena
            done.wait(10)
            return [arenas[(name, "0")], arenas[(name, "1")]]
        worlds = [None, None]

        def make(r):
            w = parallel.World.__new__(parallel.World)
            w.multiproc, w.size, w.local, w.devices = True, 2, [r], [torch.device("cpu")]
            w.inboxes, w.flags, w.counters, w._lock = {}, {}, {}, threading.Lock()
            w.buffers(("step", 7), torch.empty(3, 5, dtype=torch.bfloat16))
            worlds[r] = w
        saved = ipc.share
        ipc.share = fake_share
        try:
            threads = [threading.Thread(target=make, args=(r,), name=str(r)) for r in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(30)
        finally:
            ipc.share = saved
        a, b = worlds
        key = ("step", 7)
        self.assertEqual(sorted(a.inboxes[key]), [(0, 1), (1, 0)])
        self.assertEqual(sorted(a.counters[key]), [(0, 1), (1, 0)])
        a.inboxes[key][(0, 1)].fill_(2.0)  # rank 0's partial into rank 1's inbox
        self.assertTrue(torch.all(b.inboxes[key][(0, 1)] == 2.0))
        self.assertEqual(tuple(b.inboxes[key][(0, 1)].shape), (3, 5))
        a.flags[key][(0, 1)].fill_(1)
        self.assertEqual(int(b.flags[key][(0, 1)]), 1)
        a.counters[key][(0, 1)].fill_(5)
        self.assertEqual(int(b.counters[key][(0, 1)]), 5)
        b.flags[key][(1, 0)].fill_(1)
        b.reset_rings()  # rank 1 zeroes its own card's words: the flags raised on it and its own counters
        self.assertEqual(int(b.flags[key][(0, 1)]), 0, "a flag on card 1")
        self.assertEqual(int(b.flags[key][(1, 0)]), 1, "a flag on card 0 is rank 0's to reset")
        self.assertEqual(int(b.counters[key][(0, 1)]), 5, "rank 0's counter stays")


class SharedCommTest(unittest.TestCase):
    def test_two_processes_alias_each_others_boxes(self):
        arenas = {}
        lock = threading.Lock()
        done = threading.Barrier(2)

        def share_for(rank):
            def share(name, arena):
                with lock:
                    arenas[rank] = arena
                done.wait(10)
                return [arenas[0], arenas[1]]
            return share
        comms = [None, None]

        def make(rank):
            comms[rank] = allreduce.Comm(["cpu", "cpu"], sites=3, max_elems=64, dtypes=[torch.float32, torch.bfloat16],
                                        local=rank, share=share_for(rank))
        threads = [threading.Thread(target=make, args=(r,)) for r in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        a, b = comms
        self.assertEqual(a.world, 2)
        self.assertIs(a.arenas[0], b.arenas[0])
        self.assertIs(a.arenas[1], b.arenas[1])
        # rank 0 pushes into rank 1's inbox for source 0 at site 2: rank 1 reads it through its own Comm
        a.box(1, 2, 0, torch.float32).fill_(7.0)
        self.assertTrue(torch.all(b.box(1, 2, 0, torch.float32) == 7.0))
        self.assertIsNone(a.inbox[0][2][0], "no inbox for a card's own source")
        a.flags[1][2, 0, 0] = 1  # the flag rank 0 raises on card 1
        self.assertEqual(int(b.flags[1][2, 0, 0]), 1)
        self.assertEqual(b.gather_box(0, 1, 1, torch.bfloat16).numel() * 2, a.slice_bytes, "a slice's bytes, as bf16")
        self.assertEqual(tuple(a.counters[1].shape), (3, 3))
        self.assertEqual(tuple(a.trace[0].shape), (3, 2))
        # the regions of one card are disjoint: a store in one box never shows in another
        a.slot(0, 1, torch.float32).fill_(3.0)
        self.assertTrue(torch.all(a.box(0, 1, 1, torch.float32) == 0))
        self.assertTrue(torch.all(a.box(0, 0, 1, torch.float32) == 0))
        b.reset()  # only its own card's flags and counters
        self.assertEqual(int(b.flags[1][2, 0, 0]), 0)
        self.assertEqual(b.local, 1)

    def test_one_process_layout_unchanged(self):
        c = allreduce.Comm(["cpu", "cpu", "cpu"], sites=2, max_elems=40, dtypes=[torch.float32])
        self.assertEqual(c.local, None)
        self.assertEqual(c.box(2, 1, 0, torch.float32).numel(), c.max_bytes // 4)
        self.assertEqual(tuple(c.flags[0].shape), (2, 2, 3))
        self.assertEqual(c.gather_box(1, 0, 2, torch.float32).numel() * 4, -(-c.slice * 4 // 16) * 16)
        self.assertEqual(len(c.arenas), 3)
        self.assertTrue(all(a.device.type == "cpu" for a in c.arenas))


if __name__ == "__main__":
    unittest.main()
