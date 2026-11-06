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

"""An online acceptance estimator over the drafter's positions (hero run 15), in the vLLM fork's form.

The fork's OnlineAcceptanceEstimator (vllm/v1/worker/gpu/spec_decode/acceptance_estimator.py) predicts a draft's
acceptance from one feature, the log-odds of the drafter's maximum probability at the position: p_k = sigmoid(omega *
f + beta_k), one slope shared by the positions and an intercept per position, refit after every step from the graded
drafts (accepted or rejected per position; the bonus token is not a draft) by a damped Newton step. A row's draft k
survives with probability p_1 * ... * p_k, which ranks the slots (vxpu.engine._flat_layout under the slots policy).

Every rank keeps its own copy and grades the same rows with the same numbers in the same order, so the fits are
identical and the slot layout stays rank-identical without a broadcast. The refit is a few thousand doubles through
numpy a step, under a tenth of a millisecond.
"""
import math

import numpy as np


class OnlineAcceptanceEstimator:
    MAX_LOGODDS = 40.0  # the fitted log-odds and the intercepts kept off saturation (their constant)

    def __init__(self, positions, omega=0.5, beta=-0.2, history=1024, ridge=1.0, max_step=1.0):
        self.K = int(positions)
        self.omega0, self.beta0 = float(omega), float(beta)
        self.omega = float(omega)
        self.beta = [float(beta)] * self.K
        self.history = int(history)  # graded drafts kept per position (the fit's window)
        self.ridge = float(ridge)  # the pull toward the initial values: it damps the step where a position has few samples
        self.max_step = float(max_step)
        self.f = [[] for _ in range(self.K)]
        self.y = [[] for _ in range(self.K)]
        self.seen = [0] * self.K
        self.pending = 0

    @staticmethod
    def feature(confidence):
        """The feature of a draft: the log-odds of its probability (the drafter's maximum probability), clamped."""
        q = min(1.0 - 1e-7, max(1e-7, float(confidence)))
        return max(-40.0, min(40.0, math.log(q) - math.log1p(-q)))

    def p(self, k, confidence):
        """The predicted acceptance of a draft at position k (0-based) with this probability."""
        z = self.omega * self.feature(confidence) + self.beta[k]
        z = max(-self.MAX_LOGODDS, min(self.MAX_LOGODDS, z))
        return 1.0 / (1.0 + math.exp(-z))

    def grade(self, k, confidence, accepted):
        f, y = self.f[k], self.y[k]
        f.append(self.feature(confidence))
        y.append(1.0 if accepted else 0.0)
        if len(f) > self.history:
            del f[0]
            del y[0]
        self.seen[k] += 1
        self.pending += 1

    def grade_row(self, confidences, issued, accepted):
        """A row's step: `confidences` (probabilities, one per block position), `issued` drafts of which the first
        `accepted` were taken: those graded accepted, the first rejected one graded rejected, the rest not graded."""
        issued, accepted = int(issued), int(accepted)
        for k in range(min(issued, accepted, self.K)):
            self.grade(k, confidences[k], True)
        if accepted < min(issued, self.K):
            self.grade(accepted, confidences[accepted], False)

    def refit(self):
        """One damped Newton step on the retained graded drafts. The normal equations are an arrowhead: the slope's
        row and column couple it to every intercept, the intercepts do not couple, so the slope's step comes from
        the Schur complement and each intercept's from its own row."""
        if not self.pending:
            return
        self.pending = 0
        g_w = -self.ridge * (self.omega - self.omega0)
        h_ww = self.ridge
        g_b, h_wb, h_bb = [], [], []
        for k in range(self.K):
            if not self.f[k]:
                g_b.append(-self.ridge * (self.beta[k] - self.beta0))
                h_wb.append(0.0)
                h_bb.append(self.ridge)
                continue
            f = np.asarray(self.f[k], dtype=np.float64)
            y = np.asarray(self.y[k], dtype=np.float64)
            z = np.clip(self.omega * f + self.beta[k], -self.MAX_LOGODDS, self.MAX_LOGODDS)
            p = 1.0 / (1.0 + np.exp(-z))
            r = y - p
            w = p * (1.0 - p)
            g_w += float(np.dot(r, f))
            h_ww += float(np.dot(w, f * f))
            g_b.append(float(r.sum()) - self.ridge * (self.beta[k] - self.beta0))
            h_wb.append(float(np.dot(w, f)))
            h_bb.append(float(w.sum()) + self.ridge)
        denom = h_ww - sum(hwb * hwb / hbb for hwb, hbb in zip(h_wb, h_bb))
        num = g_w - sum(hwb * gb / hbb for hwb, gb, hbb in zip(h_wb, g_b, h_bb))
        d_w = num / denom if denom > 1e-12 else 0.0
        d_w = max(-self.max_step, min(self.max_step, d_w))
        self.omega += d_w
        for k in range(self.K):
            d_b = (g_b[k] - h_wb[k] * d_w) / h_bb[k]
            self.beta[k] += max(-self.max_step, min(self.max_step, d_b))
            self.beta[k] = max(-self.MAX_LOGODDS, min(self.MAX_LOGODDS, self.beta[k]))

    def state(self):
        return {"omega": round(self.omega, 4), "beta": [round(b, 4) for b in self.beta], "graded": list(self.seen)}
