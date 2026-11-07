"""The rank-identity check on the step's program choice (VXPU_MP_CHECK_PROGRAM=<log prefix>): engine.py of a vxpu tree."""
import sys
eng = sys.argv[1]
s = open(eng).read()
half = s.index("    def _tp_spec_half(")
tail = s[half:]
old = ("            else:\n"
       "                vrows = next((n for n in lengths if n >= max(issued) + 1), S)\n"
       "                verify, vkey = self._tp_decodes[vrows], vrows\n")
assert tail.count(old) == 1, tail.count(old)
new = old + (
    "            if _check_prog is not None:\n"
    "                # the rank-identity check (hero run 15, after F105): every rank's chosen program this step, gathered over\n"
    "                # NCCL and logged; a disagreement stops the run here rather than as an illegal address in a collective\n"
    "                _t = torch.tensor([float(T if layout is not None else vrows)], dtype=torch.float32, device=dev0)\n"
    "                _gath = _mp.allgather(_t)\n"
    "                torch.cuda.synchronize(dev0)\n"
    "                _choice = [int(v) for v in _gath.tolist()]\n"
    "                _check_log.write(f\"{j} {_choice}\\n\")\n"
    "                _check_log.flush()\n"
    "                if len(set(_choice)) > 1:\n"
    "                    raise RuntimeError(f\"program choice diverged at step {j}: ranks chose {_choice}\")\n")
tail = tail.replace(old, new)
old = "        dfin, dbuf, dev_ev = launch_draft(0)\n"
assert tail.count(old) == 1, tail.count(old)
new = ("        _check_prog = os.environ.get(\"VXPU_MP_CHECK_PROGRAM\")\n"
       "        _check_log = None\n"
       "        if _check_prog is not None:\n"
       "            from . import multiproc as _mp\n"
       "            _check_log = open(f\"{_check_prog}-r{_mp.rank()}.log\", \"a\")\n") + old
tail = tail.replace(old, new)
s = s[:half] + tail
assert "\nimport os\n" in s or "import os" in s[:3000]
open(eng, "w").write(s)
print("patched", eng)
