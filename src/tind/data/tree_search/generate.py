import numpy as np
import os
import sys
import argparse

from tind.data.tree_search.tree_search_state import TreeSearchState
from tind.data.tree_search.tree_search_spec_generator import generate_tree_search_spec
from tind.data.tree_search.utils import get_file_name, get_rewards

from tind.algorithms.selection import select_method
from tind.utils import get_random_states_train


def get_random_tree_search(num_arms, max_depth, rewards, random_state):
    assert isinstance(num_arms, int)
    assert isinstance(max_depth, int)
    assert isinstance(rewards, (list, np.ndarray))

    tree_search_spec = generate_tree_search_spec(
        num_arms=num_arms,
        max_depth=max_depth,
        rewards=rewards,
        random_state=random_state
    )

    start_state = TreeSearchState(tree_search_spec, 0, None)
    return start_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--num_instances', type=int, required=True)
    parser.add_argument('--num_goals', type=int, required=True)
    parser.add_argument('--num_arms', type=int, required=True)
    parser.add_argument('--max_depth', type=int, required=True)
    parser.add_argument('--num_simulations', type=int, required=True)
    parser.add_argument('--max_tree_depth', type=int, required=True)

    args = parser.parse_args()

    method = args.method
    num_instances = args.num_instances
    num_goals = args.num_goals
    num_arms = args.num_arms
    max_depth = args.max_depth
    num_simulations = args.num_simulations
    max_tree_depth = args.max_tree_depth

    assert num_instances > 0
    assert num_goals > 0
    assert num_arms > 0
    assert max_depth > 0
    assert num_simulations > 0
    assert max_tree_depth > 0

    seed = 42
    random_state_problem, random_state_search = get_random_states_train(seed)

    num_traces = 100
    num_rollouts = 10

    rewards = get_rewards(num_goals)

    str_directory, str_file = get_file_name(
        method, num_instances, num_goals, num_arms, max_depth, num_traces, num_simulations, max_tree_depth, num_rollouts)
    os.makedirs(str_directory, exist_ok=True)
    file_input = open(os.path.join(str_directory, str_file), 'w')
    sys.stdout = file_input

    for ind_instance in range(0, num_instances):
        start_state = get_random_tree_search(num_arms, max_depth, rewards, random_state_problem)

        for ind_trace in range(0, num_traces):
            search_node = select_method(
                method, start_state, None, num_simulations, max_tree_depth, num_rollouts, True, random_state_search, True)
            next_action = search_node.next_action()

            print('end_of_sample')

    file_input.close()