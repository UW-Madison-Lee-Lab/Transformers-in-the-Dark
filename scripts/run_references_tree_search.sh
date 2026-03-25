#!/bin/bash

METHODS='
    uls
    gls
    ups
    pps_uct
    pps_greedy
    pps_pure_exploration
'
SEEDS_BASE='1001'
NUM_GOALSS='4 8'

##

NUM_ARMSS='2'
MAX_DEPTHS='6 8'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

for METHOD in $METHODS
do
    for NUM_GOALS in $NUM_GOALSS
    do
        for NUM_ARMS in $NUM_ARMSS
        do
            for MAX_DEPTH in $MAX_DEPTHS
            do
                for SEED_BASE in $SEEDS_BASE
                do
                    echo $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE

                    source run_references_tree_search_single.sh $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE
                    sleep 1s
                done
            done
        done
    done
done

##

NUM_ARMSS='4'
MAX_DEPTHS='4'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

for METHOD in $METHODS
do
    for NUM_GOALS in $NUM_GOALSS
    do
        for NUM_ARMS in $NUM_ARMSS
        do
            for MAX_DEPTH in $MAX_DEPTHS
            do
                for SEED_BASE in $SEEDS_BASE
                do
                    echo $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE

                    source run_references_tree_search_single.sh $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE
                    sleep 1s
                done
            done
        done
    done
done

##

NUM_ARMSS='2'
MAX_DEPTHS='6'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

NUM_GOALSS='2'

for METHOD in $METHODS
do
    for NUM_GOALS in $NUM_GOALSS
    do
        for NUM_ARMS in $NUM_ARMSS
        do
            for MAX_DEPTH in $MAX_DEPTHS
            do
                for SEED_BASE in $SEEDS_BASE
                do
                    echo $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE

                    source run_references_tree_search_single.sh $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE
                    sleep 1s
                done
            done
        done
    done
done

##

NUM_ARMSS='2'
MAX_DEPTHS='7'
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=10

NUM_GOALSS='8'

for METHOD in $METHODS
do
    for NUM_GOALS in $NUM_GOALSS
    do
        for NUM_ARMS in $NUM_ARMSS
        do
            for MAX_DEPTH in $MAX_DEPTHS
            do
                for SEED_BASE in $SEEDS_BASE
                do
                    echo $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE

                    source run_references_tree_search_single.sh $METHOD $NUM_GOALS $NUM_ARMS $MAX_DEPTH $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE
                    sleep 1s
                done
            done
        done
    done
done