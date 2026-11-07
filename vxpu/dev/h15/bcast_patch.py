"""Rank 0's drafter rows for every rank's layout (VXPU_MP_BCAST_ROWS=1), their broadcast of the confidences: engine.py."""
import sys
eng = sys.argv[1]
s = open(eng).read()
half = s.index("    def _tp_spec_half(")
tail = s[half:]
old = ("        _check_prog = os.environ.get(\"VXPU_MP_CHECK_PROGRAM\")\n"
       "        _check_log = None\n"
       "        if _check_prog is not None:\n"
       "            from . import multiproc as _mp\n")
assert tail.count(old) == 1, tail.count(old)
new = ("        _check_prog = os.environ.get(\"VXPU_MP_CHECK_PROGRAM\")\n"
       "        _bcast_rows = os.environ.get(\"VXPU_MP_BCAST_ROWS\") == \"1\"\n"
       "        _check_log = None\n"
       "        if _check_prog is not None or _bcast_rows:\n"
       "            from . import multiproc as _mp\n"
       "        if _check_prog is not None:\n")
tail = tail.replace(old, new)
old = ("            dev_ev.synchronize()\n"
       "            t_drafted = time.perf_counter()\n"
       "            rows_out = dbuf.tolist()\n")
assert tail.count(old) == 1, tail.count(old)
new = ("            dev_ev.synchronize()\n"
       "            t_drafted = time.perf_counter()\n"
       "            if _bcast_rows:\n"
       "                # every rank lays the step out from rank 0's drafter rows (tokens and confidences), as their\n"
       "                # verifier broadcasts the confidences from rank 0 before its budget selection: one small gather\n"
       "                # a step, and no survival source can make the ranks disagree (hero run 15, after F105)\n"
       "                _w = dbuf.shape[1]\n"
       "                _g = _mp.allgather(dbuf.to(torch.float32))\n"
       "                torch.cuda.synchronize(dev0)\n"
       "                rows_out = [[int(round(v)) for v in r[:_w]] for r in _g.tolist()]\n"
       "            else:\n"
       "                rows_out = dbuf.tolist()\n")
tail = tail.replace(old, new)
s = s[:half] + tail
open(eng, "w").write(s)
print("patched", eng)
