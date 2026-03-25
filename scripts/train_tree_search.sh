#!/bin/bash

BLOCK_SIZE=2048
NUM_INSTANCES=200
NUM_GOALSS='4 8'

##

NUM_ARMSS='2'
MAX_DEPTHS='6 8'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

for NUM_GOALS in $NUM_GOALSS
do
    for NUM_ARMS in $NUM_ARMSS
    do
        for MAX_DEPTH in $MAX_DEPTHS
        do
            echo $BLOCK_SIZE $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH

            source train_tree_search_single.sh $BLOCK_SIZE $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH
            sleep 1s
        done
    done
done

##

NUM_ARMSS='4'
MAX_DEPTHS='4'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

for NUM_GOALS in $NUM_GOALSS
do
    for NUM_ARMS in $NUM_ARMSS
    do
        for MAX_DEPTH in $MAX_DEPTHS
        do
            echo $BLOCK_SIZE $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH

            source train_tree_search_single.sh $BLOCK_SIZE $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH
            sleep 1s
        done
    done
done