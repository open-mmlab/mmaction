#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

$PYTHON -m torch.distributed.launch --nproc_per_node=$3 $(dirname "$0")/test_recognizer.py $1 $2 --launcher pytorch ${@:4}
