#!/bin/bash

METHOD=$1
NUM_INSTANCES=$2
NUM_GOALS=$3
NUM_ARMS=$4
MAX_DEPTH=$5
NUM_SIMULATIONS=$6
MAX_TREE_DEPTH=$7

echo $METHOD $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH
echo $''

echo 'generate'
python -m tind.data.tree_search.generate --method $METHOD --num_instances $NUM_INSTANCES --num_goals $NUM_GOALS --num_arms $NUM_ARMS --max_depth $MAX_DEPTH --num_simulations $NUM_SIMULATIONS --max_tree_depth $MAX_TREE_DEPTH
echo $''
sleep 1s

echo 'prepare'
python -m tind.data.tree_search.prepare_tree_searches --method $METHOD --num_instances $NUM_INSTANCES --num_goals $NUM_GOALS --num_arms $NUM_ARMS --max_depth $MAX_DEPTH --num_simulations $NUM_SIMULATIONS --max_tree_depth $MAX_TREE_DEPTH
echo $''
sleep 1s