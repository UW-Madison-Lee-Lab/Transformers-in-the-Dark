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

##

WALL_DENSITIES_TEST='0.4 0.3 0.2 0.1'
WIDTH_HEIGHT=4
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=40

WIDTH=$WIDTH_HEIGHT
HEIGHT=$WIDTH_HEIGHT

for METHOD in $METHODS
do
    for WALL_DENSITY_TEST in $WALL_DENSITIES_TEST
    do
        for SEED_BASE in $SEEDS_BASE
        do
            echo $METHOD $WIDTH $HEIGHT $WALL_DENSITY_TEST $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE

            source run_references_navigation_single.sh $METHOD $WIDTH $HEIGHT $WALL_DENSITY_TEST $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE
            sleep 1s
        done
    done
done

##

WALL_DENSITIES_TEST='0.4'
WIDTH_HEIGHT=4
NUM_SIMULATIONSS='10 20 30 40 60 70 80 90 100'
MAX_TREE_DEPTH=40

WIDTH=$WIDTH_HEIGHT
HEIGHT=$WIDTH_HEIGHT

for METHOD in $METHODS
do
    for WALL_DENSITY_TEST in $WALL_DENSITIES_TEST
    do
        for NUM_SIMULATIONS in $NUM_SIMULATIONSS
        do
            for SEED_BASE in $SEEDS_BASE
            do
                echo $METHOD $WIDTH $HEIGHT $WALL_DENSITY_TEST $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE

                source run_references_navigation_single.sh $METHOD $WIDTH $HEIGHT $WALL_DENSITY_TEST $NUM_SIMULATIONS $MAX_TREE_DEPTH $SEED_BASE
                sleep 1s
            done
        done
    done
done