import numpy as np
import os
import sys
import argparse

from tind.data.navigation.grid_pos import GridPos
from tind.data.navigation.navigation_spec import NavigationSpec
from tind.data.navigation.navigation_state import NavigationState
from tind.data.navigation.navigation_spec_generator import generate_navigation_spec
from tind.data.navigation.navigation_spec_verifier import is_navigation_spec_solvable
from tind.data.navigation.utils import get_file_name, get_rewards

from tind.algorithms.selection import select_method
from tind.utils import get_random_states_train


def get_fixed_navigation(index=0):
    width = 5
    height = 5

    start = GridPos(0, 0)
    goal = GridPos(4, 4)

    if index == 0:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(3, 4),
        ]
    elif index == 1:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(2, 2),
            GridPos(2, 3),
            GridPos(3, 2),
            GridPos(3, 1),
            GridPos(3, 1),
        ]
    elif index == 2:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(3, 1),
            GridPos(3, 2),
            GridPos(3, 3),
        ]
    elif index == 3:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(1, 3),
            GridPos(2, 3),
            GridPos(3, 3),
        ]
    elif index == 4:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(1, 4),
            GridPos(2, 2),
            GridPos(2, 3),
            GridPos(3, 2),
        ]
    elif index == 5:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(3, 2),
            GridPos(3, 3),
        ]
    elif index == 6:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(2, 3),
            GridPos(3, 1),
            GridPos(3, 2),
            GridPos(3, 3),
        ]
    elif index == 7:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(1, 3),
            GridPos(2, 3),
            GridPos(3, 3),
        ]
    elif index == 8:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(1, 3),
            GridPos(2, 3),
            GridPos(3, 2),
            GridPos(3, 3),
        ]
    elif index == 9:
        goals = [goal]
        rewards = [1.0]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(1, 4),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(3, 3),
            GridPos(4, 3),
        ]
    elif index == 10:
        goals = [goal, GridPos(2, 2)]
        rewards = [1.0, 0.5]

        walls = [
            GridPos(1, 1),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(1, 3),
            GridPos(2, 3),
            GridPos(3, 2),
            GridPos(3, 3),
        ]
    elif index == 11:
        goals = [goal, GridPos(4, 0)]
        rewards = [1.0, 0.5]

        walls = [
            GridPos(1, 1),
            GridPos(1, 2),
            GridPos(1, 3),
            GridPos(1, 4),
            GridPos(2, 1),
            GridPos(3, 1),
            GridPos(3, 3),
            GridPos(4, 3),
        ]
    else:
        raise ValueError

    navigation_spec = NavigationSpec(width, height, start, goals, rewards, walls)
    start_state = NavigationState(navigation_spec, 0, None)

    return start_state


def get_random_navigation(width, height, wall_density, rewards, random_state):
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(wall_density, float)
    assert isinstance(rewards, (list, np.ndarray))

    solvable = False

    while not solvable:
        navigation_spec = generate_navigation_spec(width=width, height=height, wall_density=wall_density, rewards=rewards, random_state=random_state)
        solvable = is_navigation_spec_solvable(navigation_spec)

    start_state = NavigationState(navigation_spec, 0, None)
    return start_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--num_navigations', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--wall_densities', type=str, required=True)
    parser.add_argument('--num_simulations', type=int, required=True)
    parser.add_argument('--max_tree_depth', type=int, required=True)

    args = parser.parse_args()

    method = args.method
    num_navigations = args.num_navigations
    width = args.width
    height = args.height
    wall_densities = args.wall_densities
    num_simulations = args.num_simulations
    max_tree_depth = args.max_tree_depth

    list_wall_densities = wall_densities.split('-')
    assert len(list_wall_densities) > 1
    assert list_wall_densities[0] == 'wd'

    list_wall_densities = list_wall_densities[1:]
    list_wall_densities = [float(wall_density) for wall_density in list_wall_densities]

    assert num_navigations > 0
    assert width == height
    assert width > 0
    assert num_simulations > 0
    assert max_tree_depth > 0
    for wall_density in list_wall_densities:
        assert 0 <= wall_density < 1

    seed = 42
    random_state_problem, random_state_search = get_random_states_train(seed)

    num_traces = 100
    num_rollouts = 10

    num_goals = 3
    rewards = get_rewards(num_goals)

    str_directory, str_file = get_file_name(
        method, num_navigations, width, height, wall_densities, num_traces, num_simulations, max_tree_depth, num_rollouts)
    os.makedirs(str_directory, exist_ok=True)
    file_input = open(os.path.join(str_directory, str_file), 'w')
    sys.stdout = file_input

    for ind_navigation in range(0, num_navigations):
        wall_density = list_wall_densities[ind_navigation % len(list_wall_densities)]
        start_state = get_random_navigation(width, height, wall_density, rewards, random_state_problem)

        for ind_trace in range(0, num_traces):
            search_node = select_method(
                method, start_state, None, num_simulations, max_tree_depth, num_rollouts, False, random_state_search, True)
            next_action = search_node.next_action()

            print('end_of_sample')

    file_input.close()