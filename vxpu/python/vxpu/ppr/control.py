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

"""The control messages of one step, as fixed-size records (#539).

Every message is a kind word, a step counter and a stream count, then one fixed record per
stream, so a message for N streams is a known number of bytes and a mailbox slot is sized
from the most streams a step may carry (slot_bytes). The drafts of a stream go up to DEPTH
positions after its token; the mask says which of them the step issued (bit i: draft i).

  Step        coordinator to every rank: this step's token, position, drafts and mask per
              stream, and the layout the ranks run it under (the program and its rows)
  Choices     a rank to the coordinator: the token the model chose at every position of
              every stream (the verify's argmax per row), what the acceptance reads
  Acceptance  the acceptance result: per stream, how many drafts were accepted and the
              token and position the stream continues from
  Layout      the layout decision alone: the program, its rows, and each stream's first row
              and row count in it

The payload has no torch in it: ids and positions are integers, a message is bytes.
"""

import struct

DEPTH = 8  # drafts a stream may carry after its token
KIND_STEP, KIND_CHOICES, KIND_ACCEPTANCE, KIND_LAYOUT, KIND_RESET, KIND_STOP = range(1, 7)

HEAD = struct.Struct("<HHIHH")  # kind, version, step, streams, program
VERSION = 1
STEP_ROW = struct.Struct("<II" + "I" * DEPTH + "H")  # token, position, drafts[DEPTH], mask
CHOICES_ROW = struct.Struct("<" + "I" * (DEPTH + 1))  # the chosen token at each of the DEPTH + 1 positions
ACCEPT_ROW = struct.Struct("<HII")  # accepted, next token, next position
LAYOUT_ROW = struct.Struct("<HH")  # first row, rows
ROWS = {KIND_STEP: STEP_ROW, KIND_CHOICES: CHOICES_ROW, KIND_ACCEPTANCE: ACCEPT_ROW, KIND_LAYOUT: LAYOUT_ROW,
        KIND_RESET: None, KIND_STOP: None}


def slot_bytes(streams):
    """The slot capacity that holds any message of up to `streams` streams."""
    return HEAD.size + streams * max(r.size for r in ROWS.values() if r is not None)


class Message:
    """kind, step, program and the per-stream rows (tuples in the kind's record order)."""

    def __init__(self, kind, step, rows=(), program=0):
        self.kind, self.step, self.rows, self.program = kind, int(step), [tuple(r) for r in rows], int(program)

    def pack(self):
        record = ROWS[self.kind]
        out = bytearray(HEAD.pack(self.kind, VERSION, self.step, len(self.rows), self.program))
        if record is None:
            if self.rows:
                raise ValueError("control: this kind carries no rows")
            return bytes(out)
        for row in self.rows:
            out += record.pack(*row)
        return bytes(out)

    @classmethod
    def unpack(cls, payload):
        kind, version, step, streams, program = HEAD.unpack_from(payload, 0)
        if version != VERSION:
            raise ValueError(f"control: message version {version}, this code speaks {VERSION}")
        record = ROWS[kind]
        rows = [] if record is None else [record.unpack_from(payload, HEAD.size + i * record.size) for i in range(streams)]
        return cls(kind, step, rows, program)

    def __eq__(self, other):
        return isinstance(other, Message) and (self.kind, self.step, self.program, self.rows) == \
            (other.kind, other.step, other.program, other.rows)

    def __repr__(self):
        return f"Message(kind={self.kind}, step={self.step}, program={self.program}, rows={self.rows})"


def step_message(step, ids, positions, program=0):
    """A Step from each row's tokens (ids[row] = [token, draft, draft, ...], up to DEPTH + 1) and
    positions: every draft present is issued (its mask bit set), the rest padded with the token."""
    rows = []
    for toks, pos in zip(ids, positions):
        if len(toks) > DEPTH + 1:
            raise ValueError(f"control: {len(toks) - 1} drafts in a row, the schema carries {DEPTH}")
        drafts = list(toks[1:]) + [toks[0]] * (DEPTH - len(toks) + 1)
        rows.append((toks[0], pos, *drafts, (1 << (len(toks) - 1)) - 1))
    return Message(KIND_STEP, step, rows, program)


def step_rows(msg):
    """The inverse of step_message: (ids, positions) with each row's issued drafts after its token."""
    ids, positions = [], []
    for token, pos, *rest in msg.rows:
        drafts, mask = rest[:DEPTH], rest[DEPTH]
        ids.append([token] + [drafts[i] for i in range(DEPTH) if mask >> i & 1])
        positions.append(pos)
    return ids, positions


def choices_message(step, chosen):
    """Choices from each row's chosen tokens per position (padded with the last to DEPTH + 1)."""
    rows = [tuple(list(c) + [c[-1]] * (DEPTH + 1 - len(c))) for c in chosen]
    return Message(KIND_CHOICES, step, rows)
