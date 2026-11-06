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

"""The coordinator of N executor processes (#539, slice 2; the Go per-machine agent later).

It makes the shared-memory region (the step mailbox, a choices mailbox per rank, the ranks'
exchange mailboxes and the barrier), spawns one executor per rank with the plan's switches in
its environment, waits for every rank's load, and runs the step loop: one Step message a step,
the ranks' Choices back. Every rank holds the gathered logits after the head, so every rank's
choices must agree; the coordinator checks that each step and keeps rank 0's.

generate() is the engine's tensor-parallel generation loop (Engine._generate_rows_tp) over
the messages: the prompts through the step a token at a time in lockstep, then greedy decoding,
the same inputs in the same order, so its tokens are the one-process engine's.
"""

import json
import os
import subprocess
import sys
import time

from . import control, mailbox, world


class Coordinator:
    def __init__(self, artifact, size, device="cpu", env=None, exchange_bytes=8 << 20, log_dir=None, python=None):
        with open(os.path.join(artifact, "binding.json")) as f:
            binding = json.load(f)
        par = binding.get("parallel") or {}
        if par.get("world") != size:
            raise ValueError(f"{artifact} is exported for {par.get('world', 'no')} ranks, not {size}")
        self.artifact, self.size, self.device = artifact, size, device
        self.rows = int(binding.get("batch", 1))
        self.step_tokens = int(binding.get("step_tokens", 1) or 1)
        self.me = size  # the coordinator is the member after the ranks
        self.steps = 0
        boxes = {"step": mailbox.BoxSpec(self.me, range(size), control.slot_bytes(self.rows))}
        boxes.update({f"choices{r}": mailbox.BoxSpec(r, [self.me], control.slot_bytes(self.rows)) for r in range(size)})
        boxes.update(world.exchange_boxes(size, exchange_bytes))
        self.region = mailbox.Region.create(mailbox.region_name(), range(size + 1), boxes)
        spec = json.dumps(self.region.spec())
        child_env = dict(os.environ)
        child_env.update(env or {})
        # the children import the vxpu this process runs, whatever their working directory
        package_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        child_env["PYTHONPATH"] = package_root + (os.pathsep + child_env["PYTHONPATH"] if child_env.get("PYTHONPATH") else "")
        self.logs = []
        self.procs = []
        try:
            for r in range(size):
                cmd = [python or sys.executable, "-m", "vxpu.ppr.executor", "--rank", str(r), "--size", str(size),
                       "--region", spec, "--artifact", artifact, "--device", device]
                log = open(os.path.join(log_dir, f"rank{r}.log"), "w") if log_dir else None
                self.logs.append(log)
                self.procs.append(subprocess.Popen(cmd, env=child_env, stdout=log, stderr=subprocess.STDOUT if log else None))
            t0 = time.perf_counter()
            self.region.barrier.wait(self.me, check=self.check)  # every rank loaded
            self.load_seconds = time.perf_counter() - t0
        except BaseException:
            self.close(stop=False)
            raise

    def check(self):
        """Raise if an executor has exited (called while a wait yields)."""
        for r, p in enumerate(self.procs):
            if p.poll() is not None:
                log = self.logs[r].name if self.logs[r] else "its stderr"
                raise RuntimeError(f"ppr: executor rank {r} exited with {p.returncode} (see {log})")

    def reset(self):
        """Every rank's state zeroed (a new group of prompts)."""
        self.region.mailbox["step"].publish(control.Message(control.KIND_RESET, self.steps).pack())

    def step(self, ids, positions):
        """One step: ids[row] = [token, drafts...], positions[row]; returns each row's chosen token per
        position (rank 0's; every rank's must agree)."""
        self.steps += 1
        self.region.mailbox["step"].publish(control.step_message(self.steps, ids, positions).pack())
        replies = [control.Message.unpack(self.region.mailbox[f"choices{r}"].receive(self.me, check=self.check)[1])
                   for r in range(self.size)]
        for r, reply in enumerate(replies):
            if reply.step != self.steps or reply.kind != control.KIND_CHOICES:
                raise RuntimeError(f"ppr: rank {r} answered step {self.steps} with {reply}")
            if reply.rows != replies[0].rows:
                raise RuntimeError(f"ppr: rank {r}'s choices differ from rank 0's at step {self.steps}: "
                                   f"{reply.rows} against {replies[0].rows}")
        width = max(len(row) for row in ids)
        return [list(row[:width]) for row in replies[0].rows]

    def generate(self, prompts, max_new_tokens, eos_ids=()):
        """Engine._generate_rows_tp over the messages, greedy: the prompts in groups of the
        artifact's rows, each fed a token at a time in lockstep, then max_new_tokens steps."""
        rows = self.rows
        out = {"token_ids": [], "prefill_ms": [], "step_ms": []}
        for first in range(0, len(prompts), rows):
            group = [list(p) for p in prompts[first:first + rows]]
            n = len(group)
            self.reset()
            longest = max(len(p) for p in group)
            started = time.perf_counter()
            for t in range(longest):
                ids, pos = [], []
                for p in group:
                    k = t - (longest - len(p))
                    ids.append([p[max(k, 0)]])
                    pos.append(max(k, 0))
                ids += [[0]] * (rows - len(ids))
                pos += [0] * (rows - len(pos))
                chosen = self.step(ids, pos)
            out["prefill_ms"] += [(time.perf_counter() - started) * 1e3 / n] * n
            chosen = [c[0] for c in chosen]
            tokens = [[] for _ in group]
            fed = [p[-1] for p in group]
            at = [len(p) - 1 for p in group]
            done = [False] * n
            for _ in range(max_new_tokens):
                for i in range(n):
                    if not done[i]:
                        tokens[i].append(chosen[i])
                        if chosen[i] in eos_ids:
                            done[i] = True
                if all(done):
                    break
                ids, pos = [], []
                for i in range(n):
                    if not done[i]:
                        fed[i], at[i] = chosen[i], at[i] + 1
                    ids.append([fed[i]])
                    pos.append(at[i])
                ids += [[0]] * (rows - len(ids))
                pos += [0] * (rows - len(pos))
                t0 = time.perf_counter()
                chosen = [c[0] for c in self.step(ids, pos)]
                out["step_ms"].append((time.perf_counter() - t0) * 1e3)
            out["token_ids"] += tokens
        return out

    def close(self, stop=True, timeout=30.0):
        if stop and all(p.poll() is None for p in self.procs):
            try:
                self.region.mailbox["step"].publish(control.Message(control.KIND_STOP, self.steps).pack(), timeout=timeout)
            except TimeoutError:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for log in self.logs:
            if log:
                log.close()
        self.region.close(unlink=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(stop=exc[0] is None)
        return False
