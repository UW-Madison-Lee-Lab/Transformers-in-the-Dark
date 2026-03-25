#!/bin/bash

BLOCK_SIZE=$1
NUM_INSTANCES_TRAIN=$2
NUM_GOALS_TRAIN=$3
NUM_GOALS_TEST=$4
NUM_ARMS_TRAIN=$5
NUM_ARMS_TEST=$6
MAX_DEPTH=$7
NUM_SIMULATIONS_TRAIN=$8
NUM_SIMULATIONS_TEST=$9
MAX_TREE_DEPTH_TRAIN=${10}
MAX_TREE_DEPTH_TEST=${11}
TEMPERATURE=${12}

METHODS_TRAIN='
    uls
    gls
    ups
    pps_uct
    pps_greedy
    pps_pure_exploration
'
SEEDS_BASE='1001'

for METHOD_TRAIN in $METHODS_TRAIN
do
    for SEED_BASE in $SEEDS_BASE
    do
        echo $METHOD_TRAIN $BLOCK_SIZE $SEED_BASE $NUM_INSTANCES_TRAIN $NUM_GOALS_TRAIN $NUM_GOALS_TEST $NUM_ARMS_TRAIN $NUM_ARMS_TEST $MAX_DEPTH $NUM_SIMULATIONS_TRAIN $NUM_SIMULATIONS_TEST $MAX_TREE_DEPTH_TRAIN $MAX_TREE_DEPTH_TEST $TEMPERATURE

        python -m tind.sample_tree_search --method_train=$METHOD_TRAIN --block_size=$BLOCK_SIZE --seed_base=$SEED_BASE --num_instances_train=$NUM_INSTANCES_TRAIN --num_goals_train=$NUM_GOALS_TRAIN --num_goals_test=$NUM_GOALS_TEST --num_arms_train=$NUM_ARMS_TRAIN --num_arms_test=$NUM_ARMS_TEST --max_depth=$MAX_DEPTH --num_simulations_train=$NUM_SIMULATIONS_TRAIN --num_simulations_test=$NUM_SIMULATIONS_TEST --max_tree_depth_train=$MAX_TREE_DEPTH_TRAIN --max_tree_depth_test=$MAX_TREE_DEPTH_TEST --temperature=$TEMPERATURE
        sleep 1s
    done
done