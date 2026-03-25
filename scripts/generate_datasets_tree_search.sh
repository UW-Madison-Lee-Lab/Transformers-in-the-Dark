#!/bin/bash

METHODS='
    uls
    gls
    ups
    pps_uct
    pps_greedy
    pps_pure_exploration
'
NUM_INSTANCESS='200'

##

NUM_GOALSS='4 8'
NUM_ARMSS='2'
MAX_DEPTHS='6 8'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

for NUM_ARMS in $NUM_ARMSS
do
    for MAX_DEPTH in $MAX_DEPTHS
    do
        python -m tind.data.tree_search.prepare_meta --num_arms $NUM_ARMS --max_depth $MAX_DEPTH
        sleep 1s

        for NUM_GOALS in $NUM_GOALSS
        do
            for METHOD in $METHODS
            do
                for NUM_INSTANCES in $NUM_INSTANCESS
                do
                    echo $METHOD $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH

                    source generate_datasets_tree_search_single.sh $METHOD $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH
                    sleep 1s
                done
            done
        done
    done
done

##

NUM_GOALSS='4 8'
NUM_ARMSS='4'
MAX_DEPTHS='4'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

for NUM_ARMS in $NUM_ARMSS
do
    for MAX_DEPTH in $MAX_DEPTHS
    do
        python -m tind.data.tree_search.prepare_meta --num_arms $NUM_ARMS --max_depth $MAX_DEPTH
        sleep 1s

        for NUM_GOALS in $NUM_GOALSS
        do
            for METHOD in $METHODS
            do
                for NUM_INSTANCES in $NUM_INSTANCESS
                do
                    echo $METHOD $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH

                    source generate_datasets_tree_search_single.sh $METHOD $NUM_INSTANCES $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH
                    sleep 1s
                done
            done
        done
    done
done