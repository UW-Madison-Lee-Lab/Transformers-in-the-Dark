#!/bin/bash

BLOCK_SIZE=$1
NUM_INSTANCES_TRAIN=$2
WIDTH_HEIGHT=$3
WALL_DENSITIES_TRAIN=$4
WALL_DENSITY_TEST=$5
NUM_SIMULATIONS_TRAIN=$6
NUM_SIMULATIONS_TEST=$7
MAX_TREE_DEPTH_TRAIN=$8
MAX_TREE_DEPTH_TEST=$9

WIDTH=$WIDTH_HEIGHT
HEIGHT=$WIDTH_HEIGHT

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
        echo $METHOD_TRAIN $BLOCK_SIZE $SEED_BASE $NUM_INSTANCES_TRAIN $WIDTH $HEIGHT $WALL_DENSITIES_TRAIN $WALL_DENSITY_TEST $NUM_SIMULATIONS_TRAIN $NUM_SIMULATIONS_TEST $MAX_TREE_DEPTH_TRAIN $MAX_TREE_DEPTH_TEST

        python -m tind.sample_navigation --method_train=$METHOD_TRAIN --block_size=$BLOCK_SIZE --seed_base=$SEED_BASE --num_instances_train=$NUM_INSTANCES_TRAIN --width=$WIDTH --height=$HEIGHT --wall_densities_train=$WALL_DENSITIES_TRAIN --wall_density_test=$WALL_DENSITY_TEST --num_simulations_train=$NUM_SIMULATIONS_TRAIN --num_simulations_test=$NUM_SIMULATIONS_TEST --max_tree_depth_train=$MAX_TREE_DEPTH_TRAIN --max_tree_depth_test=$MAX_TREE_DEPTH_TEST
        sleep 1s
    done
done