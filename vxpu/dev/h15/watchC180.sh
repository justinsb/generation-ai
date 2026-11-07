#!/bin/bash
# ends a four-process run by exact pid once one of its ranks has died, so the chain moves on instead of spinning to the timeout
until grep -qx DONE-tpstep-C180 /root/hero/h15/r15t.txt; do
  for L in F106a-16c-deep7-slots-est-fullstack F106-16c-deep7-slots-est-fullstack; do
    if pgrep -f "[l]eague8p.py $L" > /dev/null && grep -aq "AcceleratorError\|Traceback\|dumped core\|program choice diverged" /root/hero/h15/r15league-$L-r[0-3].log 2>/dev/null; then
      sleep 20
      for p in $(pgrep -f "[l]eague8p.py $L"); do kill $p; done; sleep 10
      for p in $(pgrep -f "[l]eague8p.py $L"); do kill -9 $p; done
      echo "$(date -u +%H:%M) tp-step: watch ended $L by pid after a rank died" >> /root/hero/h15/STATUS-tp-step
    fi
  done
  sleep 15
done
