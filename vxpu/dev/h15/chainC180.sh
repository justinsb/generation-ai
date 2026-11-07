#!/bin/bash
# Pod C (Nov 7 00:19 UTC, C180 = C179 rerun after the NameError fix: the estimator flag and the constants as parameters of the half-fixed loop): (1) F106a = the estimator on D94's exact path: slots by
# survival over f112 + f128, no cost model (plan sb7a), maxprob confidences, tree 97e; the A/B against D94's 4.79 of 6.14 on the
# same program mix; text gate against F97. (2) If F106a has a result: F106 = the same with the cost model and the plan constants
# (sb7e) on tree 97f = 97e + the rank-identity check (VXPU_MP_CHECK_PROGRAM=1: every step's chosen program allgathered over
# NCCL, logged per rank, a mismatch stops the run with the step number). Waits for the drafter export started by C176.
POD=C; SUFFIX=-C; R=/root/hero/h15/r15league-C.txt
. /root/hero/h15/tpcommon3.sh
has() { grep -a -q "\"label\": \"$1\"" $R; }
EXTRA="$(grep -v '^#' /root/hero/h15/f97.env | tr '\n' ' ')"
NCCL4="NCCL_P2P_LEVEL=SYS NCCL_PROTO=LL,LL128,Simple NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=2097152 NCCL_IB_DISABLE=1 CUDA_DEVICE_MAX_CONNECTIONS=32 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_SHM_DISABLE=1"
FORMENV="VXPU_MP_WIRE=bf16 VXPU_MP_FUSED_MIX=1 $NCCL4"
KS="VXPU_FP4_GU_KS=2 VXPU_FP4_DN_KS=2 VXPU_FP4_DIRECT_MAX=48"; TP4="VXPU_EXPERT_TP=4 VXPU_FP4_DN_KS=1"; FORK6=VXPU_DEEPSEEK_V41=/mnt/ssd/deepseek_v41-h15spec7/inference; SER=VXPU_LOAD_SERIAL=1; BF16="VXPU_ALLREDUCE_PUSH_BF16=1"
CONF=VXPU_DRAFT_CONF=maxprob
MP="VXPU_PHASE_TRACE=1 $FORMENV"
TABLE=VXPU_PLAN_TABLE=/root/hero/h15/table-sweep16.json
MAIN=spec3t8-est-tp4-b16
FLATS=VXPU_VERIFY_ARTIFACTS=/mnt/ssd/spec3f112-tp4-b16,/mnt/ssd/spec3f128-tp4-b16
mp8() { local label=$1 art=$2 streams=$3 cases=$4 what=$5; shift 5; has $label && return 0; [ -e /mnt/ssd/$art/binding.json ] || { echo "$label OFF: no export at /mnt/ssd/$art $(date -u +%T)" >> $R; return 0; }; quiet; log "$label ($art, $streams streams, $cases, $TREE, four processes, the full stack, $what, stamped) $*"; local uid=/root/hero/h15/mp-uid-$label-$(date +%s) pids=""; for r in 0 1 2 3; do env $CONF PYTHONPATH=$TREE $FORK6 $KS $SER VXPU_STEP_TIMEOUT=1800 VXPU_FIRST_STEP_TIMEOUT=10800 VXPU_GPU_EXPERTS=$EP0 VXPU_MP_WORLD=4 VXPU_MP_RANK=$r VXPU_MP_UID=$uid $EXTRA "$@" timeout 14400 python -u /root/hero/h15/mp_run.py /root/hero/h15/league8p.py $label /mnt/ssd/$art $streams $cases > $H/r15league-$label-r$r.log 2>&1 & pids="$pids $!"; sleep 2; done; local rc=0; for p in $pids; do wait $p || rc=$?; done; echo "exit $rc (four processes)" >> $R; grep -a -h "^LEAGUE8\|^PHASES\|tp speculation account\|acceptance estimator\|program choice\|AOTInductor segments\|OutOfMemory\|unjoined\|Traceback\|Error\|did not finish\|rank . failed" $H/r15league-$label-r0.log | grep -a -v "NCCL INFO" | tail -40 | cut -c1-3000 >> $R; for r in 1 2 3; do grep -a -h "OutOfMemory\|Traceback\|Error\|did not finish\|rank . failed\|program choice" $H/r15league-$label-r$r.log | grep -a -v "NCCL INFO" | tail -4 | sed "s/^/rank $r: /" | cut -c1-600 >> $R; done; for p in $pids; do kill $p 2>/dev/null; done; sleep 5; }
gate() { has $1 && has $2 && python3 $H/gate.py $R $1 $2 $3 >> $R 2>&1; }
until [ -e /mnt/ssd/$MAIN/binding.json ] && ! pgrep -f "vxpu.export .* -o /mnt/ssd/$MAIN" > /dev/null; do sleep 10; done
echo "--- C178: the maxprob drafter export is in place $(date -u +%T)" >> $R
TREE=/mnt/ssd/src-tp-step97f
A=F106a-16c-deep7-slots-est-fullstack
mp8 $A $MAIN 16 code "slots by survival from the online acceptance estimator (maxprob), no cost model (D94 path), the rank-identity check on" $TP4 $BF16 $MP $TABLE VXPU_FP4_FUSE_EPI=3 VXPU_PLAN=$P/sb7a.json $FLATS VXPU_MP_CHECK_PROGRAM=/root/hero/h15/progchoice-F106a
gate F97-16c-deep7-fixed7-fullstack $A code
if has $A; then
  TREE=/mnt/ssd/src-tp-step97f
  B=F106-16c-deep7-slots-est-fullstack
  mp8 $B $MAIN 16 code "slots by survival from the estimator with the cost model and the plan constants, the rank-identity check and rank 0 rows for every rank on" $TP4 $BF16 $MP $TABLE VXPU_FP4_FUSE_EPI=3 VXPU_PLAN=$P/sb7e.json $FLATS VXPU_MP_CHECK_PROGRAM=/root/hero/h15/progchoice-F106 VXPU_MP_BCAST_ROWS=1
  gate F97-16c-deep7-fixed7-fullstack $B code; gate $A $B code
else
  echo "--- F106 not started: F106a ended without a result $(date -u +%T)" >> $R
fi
echo "tp-step: C cards free $(date -u +%T)" >> $R
echo DONE-tpstep-C180 >> /root/hero/h15/r15t.txt
