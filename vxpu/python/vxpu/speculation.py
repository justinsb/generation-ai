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

"""The speculation policy the plan names (`vxpu-plan/1`'s `speculation` section, pkg/kernelplan)
and the pure functions that apply it, for the engine's drafter path to call. Nothing here touches
a tensor: the inputs are Python lists per row, so the same functions serve the Python rows loop,
a test, and the native loop through a thin binding.

The policies (pkg/planner/speculation.go scores them against the cost model):
- fixed: every step drafts `draft` tokens; a draft is kept while it equals the verified choice
  before it (greedy acceptance, the output is what one token a step would produce);
- confidence: the drafter stops at the first draft whose top probability is under
  `min_confidence`, so a step verifies fewer token rows where the drafter is unsure;
- experts: among the drafter's candidate continuations at a position the one whose experts the
  step already reads is preferred, and the drafts are cut where they would add more than
  `max_new_experts` experts a layer beyond what the batch reads for its first tokens.

Every step's decisions feed `Stats`, which the record carries (per position: drafts issued,
accepted, and the confidence of accepted and rejected drafts), so the planner's estimates of a
policy's precision are replaced by measurements."""

import dataclasses

FIXED, CONFIDENCE, EXPERTS = "fixed", "confidence", "experts"


@dataclasses.dataclass(frozen=True)
class Policy:
    draft: int = 0  # tokens drafted a step; 0: no drafter
    policy: str = FIXED
    min_confidence: float = 0.0
    max_new_experts: int = 0
    candidates: int = 1

    @classmethod
    def from_plan(cls, section):
        """The plan file's `speculation` section as a Policy (None or {}: no drafter), checked as
        the planner checks it."""
        if not section:
            return cls()
        p = cls(draft=int(section.get("draft", 0)), policy=section.get("policy", FIXED),
                min_confidence=float(section.get("min_confidence", 0.0)),
                max_new_experts=int(section.get("max_new_experts", 0)), candidates=int(section.get("candidates", 1)))
        p.check()
        return p

    def check(self):
        if not 0 <= self.draft <= 8:
            raise ValueError(f"speculation draft {self.draft}: want 0 to 8")
        if self.policy == CONFIDENCE and not 0 < self.min_confidence < 1:
            raise ValueError(f"speculation policy {self.policy!r}: min_confidence {self.min_confidence}, want a probability in (0, 1)")
        if self.policy == EXPERTS and (self.max_new_experts < 0 or self.candidates < 1):
            raise ValueError(f"speculation policy {self.policy!r}: max_new_experts {self.max_new_experts}, candidates {self.candidates}")
        if self.policy not in (FIXED, CONFIDENCE, EXPERTS):
            raise ValueError(f"speculation policy {self.policy!r}: the engine knows {FIXED!r}, {CONFIDENCE!r} and {EXPERTS!r}")
        return self


def accept(draft_tokens, target_choices):
    """How many of a row's drafts the verify keeps: each draft that equals the choice the verify
    made at the position before it, up to the first that does not (the native loop's rule). The
    policy does not change acceptance, only what is drafted; the output stays what one token a
    step would produce."""
    n = 0
    for draft, choice in zip(draft_tokens, target_choices):
        if draft != choice:
            break
        n += 1
    return n


def draft_length(draft_probs, policy):
    """How many of the drafter's proposed tokens a row issues this step under the policy, from
    the top probability the drafter gave each (in order). fixed: all of them, up to the policy's
    draft; confidence: up to the first under min_confidence."""
    k = min(policy.draft, len(draft_probs))
    if policy.policy != CONFIDENCE:
        return k
    for i in range(k):
        if draft_probs[i] < policy.min_confidence:
            return i
    return k


def new_experts(token_experts, reading):
    """The experts a draft token adds, per layer, beyond those in `reading` (per layer, the ids
    the batch reads for its first tokens): the sum over layers of the ids not already read."""
    return sum(len(set(ids) - set(reading[layer])) for layer, ids in enumerate(token_experts))


def prefer(candidates, reading):
    """Among a position's candidate continuations, the index of the one whose experts the step
    already reads most (fewest new experts); ties go to the first, the drafter's own order (its
    most probable candidate). `candidates` is a list of per-layer expert id lists, one per
    candidate; `reading` is the per-layer ids already read."""
    best, best_new = 0, None
    for i, token_experts in enumerate(candidates):
        n = new_experts(token_experts, reading)
        if best_new is None or n < best_new:
            best, best_new = i, n
    return best


def plan_drafts(candidates_by_position, reading, policy):
    """The experts policy applied to one row's step: at each position choose the preferred
    candidate, stop where the drafts so far would add more than max_new_experts experts a layer
    on average beyond what the batch reads. Returns the chosen candidate index per issued
    position. `candidates_by_position[i]` is the list of candidates at draft position i, each a
    per-layer list of expert ids; the experts of a chosen draft count as read for the next."""
    layers = len(reading)
    budget = policy.max_new_experts * layers
    chosen, added = [], 0
    read = [set(ids) for ids in reading]
    for position, candidates in enumerate(candidates_by_position[:policy.draft]):
        if not candidates:
            break
        i = prefer(candidates, read) if policy.policy == EXPERTS else 0
        cost = new_experts(candidates[i], read)
        if policy.policy == EXPERTS and added + cost > budget:
            break
        added += cost
        chosen.append(i)
        for layer, ids in enumerate(candidates[i]):
            read[layer].update(ids)
    return chosen


class Stats:
    """Per draft position: drafts issued, accepted, and the drafter's confidence on the accepted
    and the rejected ones, for the record (`speculation` stat) and the planner's next fit."""

    def __init__(self, positions=8):
        self.issued = [0] * positions
        self.accepted = [0] * positions
        self.confidence_accepted = [0.0] * positions
        self.confidence_rejected = [0.0] * positions
        self.steps = 0

    def record(self, issued, accepted, draft_probs=None):
        """One row's step: `issued` drafts, of which the first `accepted` were kept."""
        self.steps += 1
        for i in range(issued):
            self.issued[i] += 1
            if i < accepted:
                self.accepted[i] += 1
            if draft_probs is not None and i < len(draft_probs):
                if i < accepted:
                    self.confidence_accepted[i] += draft_probs[i]
                else:
                    self.confidence_rejected[i] += draft_probs[i]

    def summary(self):
        """Per position the acceptance rate and the mean confidence of accepted and rejected
        drafts; `alpha` is the per-position rate the planner's geometric model uses (the mean
        over positions weighted by what was issued)."""
        positions = []
        for i in range(len(self.issued)):
            if not self.issued[i]:
                continue
            rejected = self.issued[i] - self.accepted[i]
            positions.append({"position": i + 1, "issued": self.issued[i], "accepted": self.accepted[i],
                              "rate": round(self.accepted[i] / self.issued[i], 3),
                              "confidence_accepted": round(self.confidence_accepted[i] / self.accepted[i], 3) if self.accepted[i] else None,
                              "confidence_rejected": round(self.confidence_rejected[i] / rejected, 3) if rejected else None})
        issued, accepted = sum(self.issued), sum(self.accepted)
        return {"steps": self.steps, "issued": issued, "accepted": accepted,
                "alpha": round(accepted / issued, 3) if issued else None, "positions": positions}
