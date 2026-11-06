#!/bin/bash
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

# The expert GEMM form's fusions (FP4_FUSE_EPI), A/B on a pod in one command (hero 15, Nov 6):
#   1. bench_fuse.py: one layer's expert node at 128 rows on synthetic TP4 slices, per kernel and as a captured
#      graph, arms 0 / 1 / 3 / 7 / 15 and the Nov 5 fold, every arm's output bit-compared to arm 0's and the direct
#      kernels'; results in $HERO/fuse-bench.log (one line and one RESULT JSON per arm).
#   2. with ARTIFACT set (a directory under /mnt/ssd): the real step through batches.py with the knob off (writes the
#      token reference) and on (MASK, default 7): the RESULT lines' match counts say the tokens are identical, their
#      step times the saving; the engine's environment (the plan, the placement, VXPU_EXPERT_TP=4 ...) is taken from
#      the caller's, as the h15 harness sets it. Logs in $HERO/fuse-step-{off,on}.log.
# Usage: HERO=/root/hero/h15 SRC=/mnt/ssd/src-fuse [ARTIFACT=spec2t7p-tp4-b1] [MASK=7] [ROWS=128] fuse-ab.sh
set -u
. $(dirname $0)/common.sh
MASK=${MASK:-7}
ROWS=${ROWS:-128}
quiet; log "fuse-bench rows=$ROWS arms=0,1,3,7,15,fold"
python -u $(dirname $0)/bench_fuse.py --rows $ROWS --arms 0,1,3,7,15,fold --graph > $HERO/fuse-bench.log 2>&1
grep -h "^rows\|^arm\|failed\|Traceback" $HERO/fuse-bench.log >> $R
unlock
if [ -n "${ARTIFACT:-}" ]; then
  REFF=$HERO/fuse-reference-$ARTIFACT.json
  rm -f $REFF
  for arm in off on; do
    [ $arm = off ] && v=0 || v=$MASK
    quiet; log "fuse-step $ARTIFACT FP4_FUSE_EPI=$v"
    VXPU_FP4_FUSE_EPI=$v python -u $(dirname $0)/batches.py --artifact /mnt/ssd/$ARTIFACT --label fuse-$arm --batches 1 \
        --mode different --prompts $(dirname $0)/prompts-code-32.txt --reference $REFF --out $OUT > $HERO/fuse-step-$arm.log 2>&1
    grep -h "RESULT\|experts:\|Traceback\|Error" $HERO/fuse-step-$arm.log | tail -6 >> $R
    unlock; sleep 10
  done
fi
