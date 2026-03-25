#!/bin/bash

METHOD=$1
WIDTH=$2
HEIGHT=$3
WALL_DENSITY_TEST=$4
NUM_SIMULATIONS=$5
MAX_TREE_DEPTH=$6
SEED_BASE=$7

python -m tind.run_references_navigation --method=$METHOD --width=$WIDTH --height=$HEIGHT --wall_density_test=$WALL_DENSITY_TEST --num_simulations=$NUM_SIMULATIONS --max_tree_depth=$MAX_TREE_DEPTH --seed_base=$SEED_BASE