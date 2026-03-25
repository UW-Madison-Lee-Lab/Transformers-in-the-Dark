#!/bin/bash

BLOCK_SIZE=$1
NUM_INSTANCES=$2
NUM_GOALS=$3
NUM_ARMS=$4
MAX_DEPTH=$5
NUM_SIMULATIONS=$6
MAX_TREE_DEPTH=$7

DATASET='tree_search'
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
        echo $WANDB_PROJECT $DATASET $METHOD $BLOCK_SIZE $SEED_BASE $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH

        python -m tind.train --wandb_project=$WANDB_PROJECT --dataset=$DATASET --method=$METHOD --block_size=$BLOCK_SIZE --seed_base=$SEED_BASE --num_instances=$NUM_INSTANCES --num_goals=$NUM_GOALS --instance_param_1=$NUM_ARMS --instance_param_2=$MAX_DEPTH --instance_param_4=$NUM_SIMULATIONS --instance_param_5=$MAX_TREE_DEPTH
    done
done