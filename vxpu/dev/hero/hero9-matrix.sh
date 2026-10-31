#!/bin/bash
# Hero run 9's acceptance table: four batches of 33 different prompts, four of 33 same prompts, four of one row,
# plus hero 8's single-batch one-row and drafter lines, each against its floor. The single-batch runs come first
# and write the token reference the batched runs are compared with.
# usage: hero9-matrix.sh LABEL [ARTS] [DRAFT_ARTS]   (artifact suffixes: "11" for rows11-b1, "m5" for specm5-b1)
L=$1; A=${2:-11}; D=${3:-m5}
. $(dirname $0)/common.sh
# single batch: the reference and hero 8's lines
bench $L-b1 rows$A-b1 1 one
bench $L-spec1 spec$D-b1 1 one
bench $L-b33 rows$A-b33 1 one
bench $L-b33 rows$A-b33 1 same
bench $L-b33 rows$A-b33 1 different
bench $L-b9 rows$A-b9 1 one
bench $L-b9 rows$A-b9 1 same
bench $L-b9 rows$A-b9 1 different
# four batches in flight: the pipeline when the engine has it, else what today's engine does with four callers
bench $L-b33x4 rows$A-b33 4 different
bench $L-b33x4 rows$A-b33 4 same
bench $L-b1x4 rows$A-b1 4 one
python3 $HERO/table.py $OUT > $HERO/table.md
echo DONE-$L >> $R
