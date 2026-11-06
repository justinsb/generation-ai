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

"""The mailboxes between processes (#539, slice 1): N fake rank processes exchange one step's
control messages 1,000 times with a coordinator, in order and with the content intact; the
median round trip is printed (the issue's target is under 50 us for four processes on a c3d).
No torch, no GPU."""

import json
import multiprocessing
import statistics
import threading
import time
import unittest

from . import control, mailbox

STREAMS = 16
STEPS = 1000


def _expected_choices(rank, msg):
    """What rank `rank` answers a Step with: per stream, the token, then each draft plus the rank
    (so a reply from the wrong rank or for the wrong step is caught), DEPTH + 1 positions."""
    rows = []
    for token, pos, *rest in msg.rows:
        drafts = rest[:control.DEPTH]
        rows.append(tuple([token + msg.step] + [d + rank + 1 for d in drafts]))
    return control.Message(control.KIND_CHOICES, msg.step, rows)


def _rank_main(spec_json, rank, n):
    """A fake executor: Step in, Choices out, n times, then the stop message and a barrier."""
    region = mailbox.Region.attach_spec(json.loads(spec_json))
    try:
        step_box, reply = region.mailbox["step"], region.mailbox[f"choices{rank}"]
        for i in range(n):
            seq, payload = step_box.receive(rank)
            msg = control.Message.unpack(payload)
            if seq != i + 1 or msg.kind != control.KIND_STEP or msg.step != i:
                raise SystemExit(f"rank {rank}: message {seq} is step {msg.step} of kind {msg.kind}, expected step {i}")
            reply.publish(_expected_choices(rank, msg).pack())
        seq, payload = step_box.receive(rank)
        if control.Message.unpack(payload).kind != control.KIND_STOP:
            raise SystemExit(f"rank {rank}: no stop after {n} steps")
        region.barrier.wait(rank)
    finally:
        region.close()


def _barrier_main(spec_json, rank, rounds):
    region = mailbox.Region.attach_spec(json.loads(spec_json))
    try:
        for _ in range(rounds):
            region.barrier.wait(rank)
    finally:
        region.close()


def _step(i):
    ids = [[(i * 7 + s * 13) % 1000 + 1] + [(i + s + d) % 500 + 2 for d in range(min(control.DEPTH, (i + s) % 9))]
           for s in range(STREAMS)]
    return control.step_message(i, ids, [i + s for s in range(STREAMS)], program=i % 3)


def _spawn(fn, args_per_child):
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=fn, args=a, daemon=True) for a in args_per_child]
    for p in procs:
        p.start()
    return procs


class MailboxTest(unittest.TestCase):
    def _round_trips(self, n_ranks):
        me = n_ranks  # the coordinator is the last participant
        boxes = {"step": mailbox.BoxSpec(me, range(n_ranks), control.slot_bytes(STREAMS), slots=4)}
        boxes.update({f"choices{r}": mailbox.BoxSpec(r, [me], control.slot_bytes(STREAMS), slots=4) for r in range(n_ranks)})
        region = mailbox.Region.create(mailbox.region_name("vxpu-t"), range(n_ranks + 1), boxes)
        spec = json.dumps(region.spec())
        procs = _spawn(_rank_main, [(spec, r, STEPS) for r in range(n_ranks)])
        trips = []
        try:
            for i in range(STEPS):
                msg = _step(i)
                t0 = time.perf_counter()
                seq = region.mailbox["step"].publish(msg.pack())
                replies = [region.mailbox[f"choices{r}"].receive(me) for r in range(n_ranks)]
                trips.append(time.perf_counter() - t0)
                self.assertEqual(seq, i + 1)
                for r, (rseq, payload) in enumerate(replies):
                    self.assertEqual(rseq, i + 1, f"rank {r}'s reply out of order")
                    self.assertEqual(control.Message.unpack(payload), _expected_choices(r, msg), f"rank {r}, step {i}")
            region.mailbox["step"].publish(control.Message(control.KIND_STOP, STEPS).pack())
            region.barrier.wait(me)
            for p in procs:
                p.join(timeout=30)
                self.assertEqual(p.exitcode, 0)
        finally:
            for p in procs:
                if p.is_alive():
                    p.kill()
            region.close(unlink=True)
        return trips

    def test_four_processes_exchange_a_step_1000_times(self):
        trips = self._round_trips(4)
        med, p90 = statistics.median(trips), sorted(trips)[int(0.9 * len(trips))]
        worst = max(range(len(trips)), key=trips.__getitem__)
        print(f"\nmailbox round trip, 4 ranks, {STREAMS} streams, {STEPS} steps: median {med * 1e6:.1f} us, "
              f"p90 {p90 * 1e6:.1f} us, max {trips[worst] * 1e6:.0f} us at step {worst} (Step out, 4 Choices back)")

    def test_two_processes_exchange_a_step(self):
        trips = self._round_trips(2)
        print(f"\nmailbox round trip, 2 ranks: median {statistics.median(trips) * 1e6:.1f} us")

    def test_barrier_over_processes(self):
        n, rounds = 4, 500
        region = mailbox.Region.create(mailbox.region_name("vxpu-b"), range(n), {})
        procs = _spawn(_barrier_main, [(json.dumps(region.spec()), r, rounds) for r in range(n)])
        try:
            for p in procs:
                p.join(timeout=60)
                self.assertEqual(p.exitcode, 0)
            gens = [mailbox.WORD.unpack_from(region.buf, region.barrier._word(r))[0] for r in range(n)]
            self.assertEqual(gens, [rounds] * n)
        finally:
            for p in procs:
                if p.is_alive():
                    p.kill()
            region.close(unlink=True)

    def test_publish_waits_for_credit(self):
        region = mailbox.Region.create(mailbox.region_name("vxpu-c"), [0, 1], {"b": mailbox.BoxSpec(0, [1], 16, slots=2)})
        try:
            box = region.mailbox["b"]
            done = threading.Event()

            def produce():
                for i in range(3):
                    box.publish(bytes([i]) * 4)
                done.set()
            threading.Thread(target=produce, daemon=True).start()
            time.sleep(0.1)
            self.assertEqual(box.published, 2, "the third message overwrote a slot nobody had read")
            self.assertEqual(box.receive(1), (1, b"\x00" * 4))
            self.assertTrue(done.wait(5))
            self.assertEqual(box.receive(1), (2, b"\x01" * 4))
            self.assertEqual(box.receive(1), (3, b"\x02" * 4))
        finally:
            region.close(unlink=True)

    def test_torn_payload_is_not_delivered(self):
        region = mailbox.Region.create(mailbox.region_name("vxpu-d"), [0, 1], {"b": mailbox.BoxSpec(0, [1], 16, slots=1)})
        try:
            box = region.mailbox["b"]
            box.publish(b"abcdefgh")
            at = box._slot(1) + mailbox.SLOT_HEADER.size
            region.buf[at] = ord("X")  # a reader that saw the sequence number before the payload landed
            with self.assertRaises(TimeoutError):
                box.receive(1, timeout=0.05)
            region.buf[at] = ord("a")
            self.assertEqual(box.receive(1), (1, b"abcdefgh"))
            box.publish(b"second")  # the credit was written: slot 1 may be reused
            self.assertEqual(box.receive(1), (2, b"second"))
        finally:
            region.close(unlink=True)

    def test_lapped_reader_is_an_error(self):
        region = mailbox.Region.create(mailbox.region_name("vxpu-e"), [0, 1], {"b": mailbox.BoxSpec(0, [1], 16, slots=1)})
        try:
            box = region.mailbox["b"]
            box.publish(b"one")
            mailbox.SLOT_HEADER.pack_into(region.buf, box._slot(2), 2, 3, 0)  # a producer that ignored the credits
            with self.assertRaises(RuntimeError):
                box.receive(1)
        finally:
            region.close(unlink=True)


class ControlTest(unittest.TestCase):
    def test_step_message_round_trips_every_depth(self):
        for depth in range(control.DEPTH + 1):
            ids = [[5] + [10 + d for d in range(depth)], [7] + [20 + d for d in range(depth)]]
            msg = control.step_message(3, ids, [11, 12], program=2)
            again = control.Message.unpack(msg.pack())
            self.assertEqual(again, msg)
            self.assertEqual(control.step_rows(again), (ids, [11, 12]))
            self.assertEqual(again.program, 2)
        with self.assertRaises(ValueError):
            control.step_message(0, [[1] + [2] * (control.DEPTH + 1)], [0])

    def test_choices_and_acceptance_pack(self):
        msg = control.choices_message(9, [[1, 2], [3]])
        self.assertEqual(control.Message.unpack(msg.pack()).rows, [(1,) + (2,) * control.DEPTH, (3,) * (control.DEPTH + 1)])
        acc = control.Message(control.KIND_ACCEPTANCE, 9, [(2, 44, 100), (0, 45, 101)])
        self.assertEqual(control.Message.unpack(acc.pack()), acc)
        self.assertEqual(len(control.Message(control.KIND_STOP, 1).pack()), control.HEAD.size)
        self.assertLessEqual(len(control.step_message(0, [[1] * 9] * 16, [0] * 16).pack()), control.slot_bytes(16))


if __name__ == "__main__":
    unittest.main()
