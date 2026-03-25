#!/bin/bash

METHODS='
    uls
    gls
    ups
    pps_uct
    pps_greedy
    pps_pure_exploration
'
NUM_NAVIGATIONSS='200'
WALL_DENSITIESS='
    wd-0.4
'

##

WIDTH_HEIGHT=4
NUM_SIMULATIONS=50
MAX_TREE_DEPTH=40

WIDTH=$WIDTH_HEIGHT
HEIGHT=$WIDTH_HEIGHT

python -m tind.data.navigation.prepare_meta --width $WIDTH --height $HEIGHT
sleep 1s

for METHOD in $METHODS
do
    for NUM_NAVIGATIONS in $NUM_NAVIGATIONSS
    do
        for WALL_DENSITIES in $WALL_DENSITIESS
        do
            echo $METHOD $NUM_NAVIGATIONS $WIDTH $HEIGHT $WALL_DENSITIES $NUM_SIMULATIONS $MAX_TREE_DEPTH

            source generate_datasets_navigation_single.sh $METHOD $NUM_NAVIGATIONS $WIDTH $HEIGHT $WALL_DENSITIES $NUM_SIMULATIONS $MAX_TREE_DEPTH
            sleep 1s
        done
    done
done