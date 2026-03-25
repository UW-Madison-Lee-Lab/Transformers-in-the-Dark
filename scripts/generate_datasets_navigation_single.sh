#!/bin/bash

METHOD=$1
NUM_MAZES=$2
WIDTH=$3
HEIGHT=$4
WALL_DENSITIES=$5
NUM_SIMULATIONS=$6
MAX_TREE_DEPTH=$7

echo $METHOD $NUM_MAZES $WIDTH $HEIGHT $WALL_DENSITIES $NUM_SIMULATIONS $MAX_TREE_DEPTH
echo $''

echo 'generate'
python -m tind.data.navigation.generate --method $METHOD --num_navigations $NUM_MAZES --width $WIDTH --height $HEIGHT --wall_densities $WALL_DENSITIES --num_simulations $NUM_SIMULATIONS --max_tree_depth $MAX_TREE_DEPTH
echo $''
sleep 1s

echo 'prepare'
python -m tind.data.navigation.prepare_navigations --method $METHOD --num_navigations $NUM_MAZES --width $WIDTH --height $HEIGHT --wall_densities $WALL_DENSITIES --num_simulations $NUM_SIMULATIONS --max_tree_depth $MAX_TREE_DEPTH
echo $''
sleep 1s