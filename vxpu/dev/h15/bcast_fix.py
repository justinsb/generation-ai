"""One line: the rows broadcast gathers a device copy of the pinned host buffer. usage: bcast_fix.py <engine.py>"""
import sys
p = sys.argv[1]
s = open(p).read()
old = "_g = _mp.allgather(dbuf.to(torch.float32))"
assert s.count(old) == 1, s.count(old)
s = s.replace(old, "_g = _mp.allgather(dbuf.to(device=dev0, dtype=torch.float32))  # dbuf is the pinned host copy")
open(p, "w").write(s)
print("patched", p)
