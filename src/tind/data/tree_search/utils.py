import numpy as np
import os

from tind.data.tree_search.tree_search_pos import TreeSearchPos


def get_path(list_path):
    list_mab_pos = [TreeSearchPos(None, 0)]

    for ind_elem, elem in enumerate(list_path):
        list_mab_pos.append(TreeSearchPos(elem, ind_elem + 1))

    return list_mab_pos


def get_file_name(method, num_instances, num_goals, num_arms, max_depth, num_traces, num_simulations, max_tree_depth, num_rollouts):
    str_postfix = f'{method}_{num_instances}_{num_goals}_{num_arms}_{max_depth}_{num_traces}_{num_simulations}_{max_tree_depth}_{num_rollouts}'

    str_directory = os.path.join('../results/data/tree_search', str_postfix)
    str_file = f'input.txt'

    return str_directory, str_file


def get_rewards(num_goals):
    if num_goals == 2:
        rewards = [1.0, 0.2]
    elif num_goals == 3:
        rewards = [1.0, 0.5, 0.2]
    elif num_goals == 4:
        rewards = [1.0, 0.5, 0.2, 0.1]
    elif num_goals == 5:
        rewards = [1.0, 0.7, 0.5, 0.2, 0.1]
    elif num_goals == 8:
        rewards = [1.0, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    elif num_goals == 10:
        rewards = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    else:
        raise ValueError

    rewards = np.array(rewards)
    assert rewards.ndim == 1
    assert rewards.shape[0] == num_goals
    return rewards