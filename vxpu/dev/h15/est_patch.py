"""Apply the acceptance-estimator change to a vxpu tree (engine.py, speculation.py) and a fork model.py.
usage: est_patch.py <vxpu package dir> <fork model.py>"""
import sys, re
pkg, model = sys.argv[1], sys.argv[2]

def edit(path, pairs):
    s = open(path).read()
    for old, new, count in pairs:
        n = s.count(old)
        assert n == count, (path, old[:60], n, count)
        s = s.replace(old, new)
    open(path, "w").write(s)
    print("patched", path)

# --- speculation.py: the plan flag
edit(f"{pkg}/speculation.py", [
    ("    cost_model: bool = False\n",
     "    cost_model: bool = False\n"
     "    # the acceptance estimator in place of the confidence head's value for the survival (hero run 15, their\n"
     "    # OnlineAcceptanceEstimator; vxpu.acceptance_estimator): p_k = sigmoid(omega * logit(q) + beta_k), refit a step\n"
     "    estimator: bool = False\n", 1),
    ('cost_model=bool(section.get("cost_model", False))',
     'cost_model=bool(section.get("cost_model", False)), estimator=bool(section.get("estimator", False))', 1),
])

# --- engine.py
eng = f"{pkg}/engine.py"
s = open(eng).read()
old = "    def _survival(rows_out, i, k, block, scale):\n"
assert s.count(old) == 1
s = s.replace(old,
    "    def _survival(rows_out, i, k, block, scale, estimator=None):\n")
old = ("        p = 1.0\n"
       "        for q in range(k + 1):\n"
       "            p *= min(1.0, max(0.0, rows_out[i][block + q] / scale)) if scale else 1.0\n"
       "        return p\n")
assert s.count(old) == 1
s = s.replace(old,
    "        p = 1.0\n"
    "        for q in range(k + 1):\n"
    "            if estimator is not None and scale:  # the fitted acceptance from the draft's probability (their estimator)\n"
    "                p *= estimator.p(q, rows_out[i][block + q] / scale)\n"
    "            else:\n"
    "                p *= min(1.0, max(0.0, rows_out[i][block + q] / scale)) if scale else 1.0\n"
    "        return p\n")
old = "self._survival(rows_out, i, k, block, scale), -i, i, k)"
assert s.count(old) == 1
s = s.replace(old, "self._survival(rows_out, i, k, block, scale, estimator), -i, i, k)")
# the _flat_layout signature: add estimator=None after account=None (either with or without constants)
m = re.search(r"def _flat_layout\(self, issued, rows_out, live, flat_sizes, fill, slots, R, S, max_draft, block, scale, cost_model=False,\n\s+account=None(, constants=None)?\):", s)
assert m, "_flat_layout signature"
sig = m.group(0)
s = s.replace(sig, sig[:-2] + ", estimator=None):")
# the closure in _tp_spec_half: build the estimator and pass it
i0 = s.index("        def flat_layout(issued, rows_out, live):\n")
j0 = s.index("\n\n", i0)
closure = s[i0:j0]
assert "account=account" in closure
closure_new = ("        from . import acceptance_estimator as _ae\n"
               "        est = _ae.OnlineAcceptanceEstimator(max_draft) if getattr(policy, \"estimator\", False) else None\n"
               + closure.replace("account=account", "account=account, estimator=est"))
s = s[:i0] + closure_new + s[j0:]
# grading: the host path and the ring readback
old = "                    req_hist[i].append(n)\n"
assert s.count(old) == 1
s = s.replace(old, old +
    "                    if est is not None:\n"
    "                        est.grade_row([v / scale for v in rows_out[i][block:2 * block]], issued[i], n)\n")
old = "                req_hist[i].append(int(n))  # the per-request controller's history"
assert s.count(old) == 1
s = s.replace(old,
    "                if est is not None:\n"
    "                    est.grade_row([v / scale for v in rows_out[i][block:2 * block]], int(iss), int(n))\n" + old)
# the refit after each step's grading: before the rows_active append that follows each grading loop in _tp_spec_half
half = s.index("    def _tp_spec_half(")
tail = s[half:]
old = "\n                out[\"rows_active\"].append(active)\n"
assert tail.count(old) == 1, tail.count(old)
tail = tail.replace(old, "\n                if est is not None:\n                    est.refit()" + old)
old = "\n            out[\"rows_active\"].append(active)\n"
assert tail.count(old) == 1, tail.count(old)
tail = tail.replace(old, "\n            if est is not None:\n                est.refit()" + old)
s = s[:half] + tail
# the state at the end of the loop: the first 'return out' after the ring readback in _tp_spec_half
half = s.index("    def _tp_spec_half(")
k = s.index("\n        return out", half)
s = s[:k] + ("\n        if est is not None:\n"
             "            out[\"estimator\"] = est.state()\n"
             "            print(f\"[vxpu] acceptance estimator: {est.state()}\")") + s[k:]
open(eng, "w").write(s)
print("patched", eng)

# --- the fork's model.py: the draft's maximum probability as the confidence (VXPU_DRAFT_CONF=maxprob)
s = open(model).read()
old = "import torch.distributed as dist\n"
assert s.count(old) == 1
s = s.replace(old, old + "import os as _os\n\n"
    "# vxpu: the drafter's confidence as the log-odds of its maximum probability at each block position (hero run 15;\n"
    "# the fork's acceptance estimator's feature) in place of the confidence head's score: VXPU_DRAFT_CONF=maxprob.\n"
    "_conf_maxprob = _os.environ.get(\"VXPU_DRAFT_CONF\", \"\") == \"maxprob\"\n\n\n"
    "def _pair_with_lse(pair, logits_i, bias):\n"
    "    \"\"\"The exchanged pair with the rank's log-sum-exp of the position's (biased) logits in its third float.\"\"\"\n"
    "    z = logits_i.float() if bias is None else logits_i.float() + bias\n"
    "    lse = torch.logsumexp(z, dim=-1)\n"
    "    return torch.cat([pair[:, :2], lse.unsqueeze(-1).to(pair.dtype), pair[:, 3:]], dim=-1)\n\n\n"
    "def _maxprob_logodds(stacked):\n"
    "    \"\"\"From the gathered pairs [world, b, 4]: the chosen token's logit (the max over the ranks' best values) less the\n"
    "    global log-sum-exp (over the ranks' partials) is log q(m*); returned as its log-odds, clamped to +-40.\"\"\"\n"
    "    m = stacked[..., 0].max(dim=0).values\n"
    "    lse = torch.logsumexp(stacked[..., 2].float(), dim=0)\n"
    "    logq = (m.float() - lse).clamp(max=-1e-7)\n"
    "    return (logq - torch.log(-torch.expm1(logq))).clamp(-40.0, 40.0)\n")
old = ("                pairs = [torch.empty_like(pair) for _ in range(world_size)]\n"
       "                dist.all_gather(pairs, pair)  # [world][b, 4]\n"
       "                stacked = torch.stack(pairs, dim=0)  # [world, b, 4]\n")
assert s.count(old) == 2
first = s.index(old)
fused = ("                if _conf_maxprob:  # the kernel left the logits slice unbiased: the bias again for the partial\n"
         "                    _w = self.markov_head.head.weight\n"
         "                    pair = _pair_with_lse(pair, logits[:, i], markov_embed.float() @ _w.float().t())\n"
         + old + "                if _conf_maxprob:\n                    logodds.append(_maxprob_logodds(stacked))\n")
s = s[:first] + fused + s[first + len(old):]
second = s.index(old, first + len(fused))
plain = ("                if _conf_maxprob:  # the bias is in the slice already\n"
         "                    pair = _pair_with_lse(pair, logits[:, i], None)\n"
         + old + "                if _conf_maxprob:\n                    logodds.append(_maxprob_logodds(stacked))\n")
s = s[:second] + plain + s[second + len(old):]
old = "        markov_embeds = []\n"
assert s.count(old) == 1
s = s.replace(old, old + "        logodds = []\n")
old = "        confidence = self.confidence_head(x, markov_embed)\n"
assert s.count(old) == 1
s = s.replace(old,
    "        if _conf_maxprob and logodds:\n"
    "            confidence = torch.stack(logodds, dim=1)  # [b, block]: the log-odds of each draft's maximum probability\n"
    "        else:\n"
    "            confidence = self.confidence_head(x, markov_embed)\n")
open(model, "w").write(s)
print("patched", model)
