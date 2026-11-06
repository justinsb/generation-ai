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

"""The executor process of one rank (#539, slice 2).

    python -m vxpu.ppr.executor --rank R --size N --region SPEC --artifact DIR [--device cpu]

It attaches the coordinator's region, makes the process world for its rank, loads that
rank's state through the engine (Engine(world=...): the engine builds world.local's rank
alone) and, after a barrier that tells the coordinator every rank is loaded, runs its step on
every Step message: the rows' tokens and positions in, the rank's step graph run (its
collectives cross the processes through the world), and its choices (the argmax at every
position of every row) published back. Reset zeroes the rank's state; Stop ends the loop.

What the engine still does per process that a per-rank load would not (the gaps, with the
lines in the tree this was written on): it fetches every dense weight whole and slices
afterwards (engine.py _fetch_bound and _tp_slice: the slice should be read from the
checkpoint), it loads the whole expert banks and n-gram tables in every process (ops.py's CPU
expert path has rank 0 compute every pair and the other ranks add zeros), and the plan's
golden check would run a whole generation inside every load, so the executor turns it off
(VXPU_CHECK_GOLDEN=0): checking tokens is the coordinator's job.
"""

import argparse
import json
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--region", required=True, help="the region spec as JSON (mailbox.Region.spec())")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    # the plan's switches this process runs under: the tensor-parallel world of the coordinator's size,
    # the dataflow runner (the CPU tensor-parallel path), no golden generation inside the load
    os.environ["VXPU_DENSE_TP"] = str(args.size)
    os.environ.setdefault("VXPU_DATAFLOW", "1")
    os.environ.setdefault("VXPU_CHECK_GOLDEN", "0")
    import torch
    from .. import engine as engine_mod
    from . import control, mailbox, world as world_mod

    tag = f"[vxpu] ppr rank {args.rank}"
    region = mailbox.Region.attach_spec(json.loads(args.region))
    try:
        world = world_mod.ProcessWorld(region, args.rank, [torch.device(args.device)] * args.size)
        t0 = time.perf_counter()
        engine = engine_mod.Engine(args.artifact, device=args.device, world=world)
        print(f"{tag}: loaded rank {args.rank} of {args.size} on {args.device} in {time.perf_counter() - t0:.1f} s "
              f"({len(engine._tp_graphs)} rank program)", flush=True)
        region.barrier.wait(args.rank)  # every rank loaded; the coordinator waits here too
        step_box, out = region.mailbox["step"], region.mailbox[f"choices{args.rank}"]
        steps = 0
        while True:
            _seq, payload = step_box.receive(args.rank)
            msg = control.Message.unpack(payload)
            if msg.kind == control.KIND_STOP:
                break
            if msg.kind == control.KIND_RESET:
                for state in engine._tp_states:
                    for t in state.values():
                        t.zero_()
                continue
            if msg.kind != control.KIND_STEP:
                raise SystemExit(f"{tag}: message of kind {msg.kind} is not a step")
            ids, positions = control.step_rows(msg)
            width = max(len(row) for row in ids)
            rows = [row + [row[0]] * (width - len(row)) for row in ids]  # a row's spare positions repeat its token
            with torch.no_grad():
                logits = engine._decode_run(input_ids=torch.tensor(rows, dtype=torch.long),
                                            cache_position=torch.tensor(positions, dtype=torch.long))
            chosen = logits.float().argmax(dim=-1)
            chosen = chosen.unsqueeze(1) if chosen.dim() == 1 else chosen  # [rows, positions]
            out.publish(control.choices_message(msg.step, chosen.tolist()).pack())
            steps += 1
        print(f"{tag}: {steps} steps, {world.exchanges} exchanges, stopping", flush=True)
    finally:
        region.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
