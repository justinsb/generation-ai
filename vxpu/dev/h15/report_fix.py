"""The estimator's end-of-run report goes at the end of _tp_spec_half (it had landed at the end of _generate_rows_tp,
which returns early on the speculative path, so no run printed it). usage: report_fix.py <engine.py>"""
import sys, re
p = sys.argv[1]
s = open(p).read()
block = ("        if est is not None:\n"
         "            out[\"estimator\"] = est.state()\n"
         "            print(f\"[vxpu] acceptance estimator: {est.state()}\")\n")
assert s.count(block) == 1, s.count(block)
s = s.replace(block, "")
start = s.index("    def _tp_spec_half(")
m = re.compile(r"\n    def ").search(s, start + 10)
end = m.start() + 1  # the start of the next method's line
body = s[start:end].rstrip("\n") + "\n"
tail = s[start + len(body):end]  # blank lines between the methods
new_body = body + ("        if est is not None:  # the fitted estimator at the end of the loop (hero run 15)\n"
                   "            out[\"estimator\"] = est.state()\n"
                   "            print(f\"[vxpu] acceptance estimator: {est.state()}\", flush=True)\n")
s = s[:start] + new_body + tail + s[end:]
open(p, "w").write(s)
print("patched", p)
