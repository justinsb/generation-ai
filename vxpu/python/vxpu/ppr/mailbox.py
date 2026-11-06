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

"""Mailboxes over shared memory: the control plane between the processes of one step (#539).

A region is one shared-memory segment whose layout both sides compute from the same spec, so
nothing in it has to be parsed: a barrier over its participants and a list of mailboxes. A
mailbox has one producer and a fixed set of consumers, a ring of fixed-size slots and a credit
word per consumer:

  publish   the producer waits until every consumer has consumed the message that last used
            the slot (the ring never laps a reader), writes the payload, its length and its
            CRC, then the slot's sequence number, which is the signal
  receive   a consumer waits for its next sequence number in the slot it expects, copies the
            payload, checks the CRC and that the sequence number still stands, then writes its
            credit word: the slot may be rewritten

Every word in the region has one writer, so the protocol needs no atomics and no lock: the
barrier is a generation word per member (a member raises its own and waits for the others'),
the credits are a word per consumer, the signal is a word per slot. Python gives no store
ordering between the payload and the sequence word, which is what the CRC is for: it is seeded
with the sequence number and written before that number, so a reader that sees the new number
over a stale or torn payload, or over the previous message's length and CRC, fails the check
and reads again (the first run of the test caught exactly that: a 16-byte header read torn
between the new sequence number and the slot's zero length, and the CRC of nothing is 0). The native data path writes these words with release and acquire stores instead; the
layout is the same.

Waiting spins for a while (a step's message arrives within microseconds) and then yields the
processor until the timeout, so an idle executor does not hold a core for ever.
"""

import mmap
import os
import struct
import time
import zlib
from multiprocessing import shared_memory

LINE = 64  # the cache line: every word that one party writes and others poll has a line of its own
SLOT_HEADER = struct.Struct("<QII")  # sequence number, then payload length and CRC32 (LEN_CRC)
LEN_CRC = struct.Struct("<II")
WORD = struct.Struct("<Q")
SPIN_SECONDS = float(os.environ.get("VXPU_PPR_SPIN", "0.002"))  # spinning before the first yield
TIMEOUT = float(os.environ.get("VXPU_PPR_TIMEOUT", "600"))  # a wait longer than this is a dead peer


class BoxSpec:
    """A mailbox: its producer, its consumers (participant indices), slots of `capacity` payload bytes."""

    def __init__(self, producer, consumers, capacity, slots=2):
        if slots < 1 or capacity < 1:
            raise ValueError("mailbox: a box needs a slot and a byte")
        self.producer = int(producer)
        self.consumers = tuple(int(c) for c in consumers)
        self.capacity = int(capacity)
        self.slots = int(slots)
        if self.producer in self.consumers:
            raise ValueError("mailbox: a box's producer does not consume it")

    def to_json(self):
        return {"producer": self.producer, "consumers": list(self.consumers), "capacity": self.capacity, "slots": self.slots}

    @classmethod
    def from_json(cls, d):
        return cls(d["producer"], d["consumers"], d["capacity"], d.get("slots", 2))

    @property
    def slot_bytes(self):
        return _round(SLOT_HEADER.size + self.capacity)

    @property
    def nbytes(self):
        return len(self.consumers) * LINE + self.slots * self.slot_bytes


def _round(n):
    return (n + LINE - 1) // LINE * LINE


def _crc(seq, payload):
    return zlib.crc32(payload, zlib.crc32(WORD.pack(seq)))


def _wait(ready, timeout, what, check=None):
    """Spin on ready() until it returns a value that is not None; yield after SPIN_SECONDS; raise after
    timeout. `check()` runs while yielding and may raise (a peer process that died)."""
    timeout = TIMEOUT if timeout is None else timeout
    v = ready()
    if v is not None:
        return v
    t0 = time.perf_counter()
    while True:
        v = ready()
        if v is not None:
            return v
        elapsed = time.perf_counter() - t0
        if elapsed > timeout:
            raise TimeoutError(f"mailbox: {what} did not arrive in {timeout:.1f} s")
        if elapsed > SPIN_SECONDS:
            if check is not None:
                check()
            time.sleep(0)


class Barrier:
    """A generation word per member: wait(me) raises mine and returns when every member's is as high."""

    def __init__(self, buf, offset, members):
        self.buf, self.offset, self.members = buf, offset, tuple(members)
        self._gen = {}  # my generation per member identity this process acts as

    def _word(self, m):
        return self.offset + self.members.index(m) * LINE

    def wait(self, me, timeout=None, check=None):
        gen = self._gen.get(me, 0) + 1
        self._gen[me] = gen
        WORD.pack_into(self.buf, self._word(me), gen)
        others = [self._word(m) for m in self.members if m != me]

        def ready():
            return True if all(WORD.unpack_from(self.buf, o)[0] >= gen for o in others) else None
        _wait(ready, timeout, f"barrier generation {gen} for member {me}", check)
        return gen

    def reset(self):
        for m in self.members:
            WORD.pack_into(self.buf, self._word(m), 0)
        self._gen.clear()


class Mailbox:
    """One producer's ring of slots, read by its consumers (BoxSpec)."""

    def __init__(self, buf, offset, spec):
        self.buf, self.offset, self.spec = buf, offset, spec
        self._slots = offset + len(spec.consumers) * LINE
        self.published = 0  # the producer's count (one producer process)
        self._received = {}  # a consumer's count, per consumer identity this process acts as

    def _credit(self, consumer):
        return self.offset + self.spec.consumers.index(consumer) * LINE

    def _slot(self, seq):
        return self._slots + ((seq - 1) % self.spec.slots) * self.spec.slot_bytes

    def publish(self, payload, timeout=None):
        """Write payload as the next message; returns its sequence number (from 1)."""
        if len(payload) > self.spec.capacity:
            raise ValueError(f"mailbox: {len(payload)} bytes into slots of {self.spec.capacity}")
        seq = self.published + 1
        freed = seq - self.spec.slots  # the message that last used this slot must be consumed by everyone
        if freed > 0:
            credits = [self._credit(c) for c in self.spec.consumers]

            def ready():
                return True if all(WORD.unpack_from(self.buf, o)[0] >= freed for o in credits) else None
            _wait(ready, timeout, f"credit for message {seq} (slot of message {freed})")
        at = self._slot(seq)
        body = at + SLOT_HEADER.size
        self.buf[body:body + len(payload)] = payload
        LEN_CRC.pack_into(self.buf, at + WORD.size, len(payload), _crc(seq, payload))
        WORD.pack_into(self.buf, at, seq)  # the signal, last
        self.published = seq
        return seq

    def receive(self, me, timeout=None, check=None):
        """The next message for consumer `me`: (sequence number, payload bytes)."""
        seq = self._received.get(me, 0) + 1
        at = self._slot(seq)
        body = at + SLOT_HEADER.size

        def ready():
            got = WORD.unpack_from(self.buf, at)[0]
            if got < seq:
                return None
            if got > seq:
                raise RuntimeError(f"mailbox: consumer {me} expected message {seq} and found {got}: lapped")
            n, crc = LEN_CRC.unpack_from(self.buf, at + WORD.size)
            payload = bytes(self.buf[body:body + n])
            if _crc(seq, payload) != crc or WORD.unpack_from(self.buf, at)[0] != seq:
                return None  # a stale or torn payload or header under the new sequence number: read again
            return payload
        payload = _wait(ready, timeout, f"message {seq} for consumer {me}", check)
        self._received[me] = seq
        WORD.pack_into(self.buf, self._credit(me), seq)
        return seq, payload

    def reset(self):
        self.buf[self.offset:self.offset + self.spec.nbytes] = bytes(self.spec.nbytes)
        self.published = 0
        self._received.clear()


class Region:
    """A shared-memory segment holding a barrier over `members` and the mailboxes of `boxes`
    ({name: BoxSpec}). create() makes it (the coordinator), attach() opens it (an executor) from the
    same spec; the layout is a function of the spec, in this order, 64-byte aligned."""

    def __init__(self, name, members, boxes, shm):
        self.name, self.members, self.boxes, self.shm = name, tuple(members), dict(boxes), shm
        self.buf = shm.buf
        offset = LINE  # the first line stays zero (a reader of an unmapped region sees nothing)
        self.barrier = Barrier(self.buf, offset, self.members)
        offset += len(self.members) * LINE
        self.mailbox = {}
        for box, spec in self.boxes.items():
            self.mailbox[box] = Mailbox(self.buf, offset, spec)
            offset += spec.nbytes
        self.nbytes = offset

    @staticmethod
    def size_of(members, boxes):
        return _round(LINE + len(members) * LINE + sum(s.nbytes for s in boxes.values()))

    @classmethod
    def create(cls, name, members, boxes):
        size = max(cls.size_of(members, boxes), mmap.PAGESIZE)
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        shm.buf[:size] = bytes(size)
        return cls(name, members, boxes, shm)

    @classmethod
    def attach(cls, name, members, boxes):
        kwargs = {"track": False} if "track" in shared_memory.SharedMemory.__init__.__code__.co_varnames else {}
        return cls(name, members, boxes, shared_memory.SharedMemory(name=name, create=False, **kwargs))

    def spec(self):
        """The spec attach() needs, as JSON-ready data."""
        return {"name": self.name, "members": list(self.members), "boxes": {k: v.to_json() for k, v in self.boxes.items()}}

    @classmethod
    def attach_spec(cls, spec):
        return cls.attach(spec["name"], spec["members"], {k: BoxSpec.from_json(v) for k, v in spec["boxes"].items()})

    def close(self, unlink=False):
        self.buf = None
        for m in self.mailbox.values():
            m.buf = None
        self.barrier.buf = None
        self.shm.close()
        if unlink:
            self.shm.unlink()


def region_name(prefix="vxpu-ppr"):
    """A name no other region on this machine has: the prefix, the pid and the clock."""
    return f"{prefix}-{os.getpid()}-{time.monotonic_ns() % 10**9}"
