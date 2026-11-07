"""estimator and the plan constants reach the half-fixed loop as parameters (policy is not in its scope): engine.py."""
import sys
eng = sys.argv[1]
s = open(eng).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:70], s.count(old))
    s = s.replace(old, new)
rep("                      cost_model=False, request=False):\n",
    "                      cost_model=False, request=False, estimator=False, constants=None):\n")
rep("                                           request=flat and policy.policy == speculation.REQUEST)\n",
    "                                           request=flat and policy.policy == speculation.REQUEST,\n"
    "                                           estimator=flat and bool(getattr(policy, \"estimator\", False)),\n"
    "                                           constants=policy.constants() if flat else None)\n")
rep("        est = _ae.OnlineAcceptanceEstimator(max_draft) if getattr(policy, \"estimator\", False) else None\n",
    "        est = _ae.OnlineAcceptanceEstimator(max_draft) if estimator else None\n")
rep("cost_model=cost_model, account=account, estimator=est)",
    "cost_model=cost_model, account=account, estimator=est,\n                                     constants=constants)")
open(eng, "w").write(s)
print("patched", eng)
