#!/bin/bash

MODEL=$1
DATA_DIR=$2
WORK_DIR=$3
PY_ARGS=${@:4}

if ! [ -d "$WORK_DIR" ]; then
   mkdir -p $WORK_DIR
fi

python benchmarks/cns_benchmark.py --model ${MODEL} --data_dir ${DATA_DIR} --output_dir ${WORK_DIR} ${PY_ARGS}
