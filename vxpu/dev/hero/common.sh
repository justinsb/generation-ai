#!/bin/bash
# Shared by the hero scripts on a pod: the engine's environment, the placement, and one process per pod.
# A step measurement needs an idle pod: another engine loading, a cache read, or any other benchmark process on the
# GPUs turns steps into flat 30 or 60 ms medians. A lock directory (mkdir is atomic) is held for the whole run, and
# the GPUs and process list are checked before starting; a lock whose owner is gone is taken over.
# Set before sourcing: SRC (the code's directory on the pod), FORK (the fork's inference directory), HERO (results dir).
. /mnt/ssd/env.sh
export VXPU_DEEPSEEK_V41=${FORK:-/mnt/ssd/deepseek_v41-h8/inference}
export PYTHONPATH=${SRC:-/mnt/ssd/src-h8}
HERO=${HERO:-/root/hero/hero9}
mkdir -p $HERO
cd /mnt/ssd
R=$HERO/results.txt
OUT=$HERO/results.jsonl
REF=$HERO/reference.json
SHARD="VXPU_GPU_EXPERTS=/mnt/ssd/placement-ranges4c.json VXPU_LAYER_DEVICES=0:8=cuda:0,8:20=cuda:1,20:32=cuda:2,32:43=cuda:3"
LOCK=/root/hero/gpu.lock
log() { echo "--- $* $(date -u +%T) load $(cut -d' ' -f1-3 /proc/loadavg)" >> $R; }
unlock() { [ "$(cat $LOCK/pid 2>/dev/null)" = "$$" ] && rm -rf $LOCK; }
trap unlock EXIT
quiet() {
  for k in $(seq 1 360); do
    if mkdir $LOCK 2>/dev/null; then echo $$ > $LOCK/pid; else
      owner=$(cat $LOCK/pid 2>/dev/null); if [ -n "$owner" ] && ! kill -0 $owner 2>/dev/null; then rm -rf $LOCK; fi; sleep 5; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    others=$(ps -eo pid,args | grep -E "[h]ero-spec|[h]ero-kprof|[k]prof-spec|[v]xpu.plan|[b]atches.py" | grep -v " $$ " | wc -l)
    if [ "$used" -lt 100 ] && [ "$others" -eq 0 ]; then return 0; fi
    rm -rf $LOCK; sleep 5
  done
  echo "NOT QUIET after 30 min" >> $R
}
# bench LABEL ARTIFACT BATCHES MODE [STRATEGY] [extra env...]: one engine load, one group of batches, results appended
bench() {
  local label=$1 art=$2 batches=$3 mode=$4; shift 4
  local strategy=auto; case "$1" in auto|sequential|threads|pipeline) strategy=$1; shift;; esac
  quiet; log "$label $art batches=$batches mode=$mode strategy=$strategy $*"
  env $SHARD "$@" python -u $HERO/batches.py --artifact /mnt/ssd/$art --label $label --batches $batches --mode $mode \
      --strategy $strategy --prompts $HERO/prompts-32.txt --reference $REF --out $OUT > $HERO/$label-$mode.log 2>&1
  grep -h "RESULT\|SAME\|Traceback\|Error" $HERO/$label-$mode.log | tail -8 >> $R
  unlock; sleep 10
}
