#!/bin/bash

METHOD=$1
NUM_GOALS=$2
NUM_ARMS=$3
MAX_DEPTH=$4
NUM_SIMULATIONS=$5
MAX_TREE_DEPTH=$6
SEED_BASE=$7

python -m tind.run_references_tree_search --method=$METHOD --num_goals=$NUM_GOALS --num_arms=$NUM_ARMS --max_depth=$MAX_DEPTH --num_simulations=$NUM_SIMULATIONS --max_tree_depth=$MAX_TREE_DEPTH --seed_base=$SEED_BASE