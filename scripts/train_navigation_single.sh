#!/bin/bash

BLOCK_SIZE=$1
NUM_INSTANCES=$2
WIDTH_HEIGHT=$3
WALL_DENSITIES=$4
NUM_SIMULATIONS=$5
MAX_TREE_DEPTH=$6

WIDTH=$WIDTH_HEIGHT
HEIGHT=$WIDTH_HEIGHT

DATASET='navigation'
WANDB_PROJECT='tind-'$DATASET
METHODS='
    uls
    gls
    ups
    pps_uct
    pps_greedy
    pps_pure_exploration
'
SEEDS_BASE='1001'

for METHOD in $METHODS
do
    for SEED_BASE in $SEEDS_BASE
    do
        echo $WANDB_PROJECT $DATASET $METHOD $BLOCK_SIZE $SEED_BASE $NUM_INSTANCES $WIDTH $HEIGHT $WALL_DENSITIES $NUM_SIMULATIONS $MAX_TREE_DEPTH

        python -m tind.train --wandb_project=$WANDB_PROJECT --dataset=$DATASET --method=$METHOD --block_size=$BLOCK_SIZE --seed_base=$SEED_BASE --num_instances=$NUM_INSTANCES --instance_param_1=$WIDTH --instance_param_2=$HEIGHT --instance_param_3=$WALL_DENSITIES --instance_param_4=$NUM_SIMULATIONS --instance_param_5=$MAX_TREE_DEPTH
    done
done