import numpy as np
import os
import io
import sys
import argparse

from tind.utils import get_random_states_test

from tind.algorithms.selection import select_method
from tind.algorithms.utils import get_search_method_best_child_method

from tind.data.navigation.generate import get_random_navigation
from tind.data.navigation.utils import get_rewards
from tind.data.navigation.navigation_spec_verifier import count_steps_to_goals


def get_results(method, width, height, wall_density_test, rewards, num_simulations, max_tree_depth, seed_base):
    assert width == height
    assert width > 0
    assert num_simulations > 0
    assert max_tree_depth > 0
    assert 0 <= wall_density_test < 1

    search_method, best_child_method = get_search_method_best_child_method(method)
    random_state_problem, random_state_search = get_random_states_test(seed_base)

    num_navigations_test = 10
    num_traces = 100
    num_rollouts = 10

    count = 0
    count_goal = 0
    list_num_tokens = []
    list_simulation_goal = []
    list_minimum_depth_goal = []
    list_steps_to_goals = []
    list_rewards_achieved = []
    list_jump_distances = []

    for ind_navigation in range(0, num_navigations_test):
        start_state = get_random_navigation(width, height, wall_density_test, rewards, random_state_problem)
        steps_to_goals = count_steps_to_goals(start_state.navigation_spec)

        for ind_trace in range(0, num_traces):
            old_stdout = sys.stdout
            my_stdout = sys.stdout = io.StringIO()

            count += 1

            search_node = select_method(
                method, start_state, None, num_simulations, max_tree_depth, num_rollouts, False, random_state_search, True)
            next_action = search_node.next_action()

            sys.stdout = old_stdout
            out = my_stdout.getvalue()

            tokens = out.split()
            num_tokens = len(tokens)

            list_num_tokens.append(num_tokens)

            simulation_goal = search_node.simulation_achieving_goal
            minimum_depth_goal = search_node.minimum_depth_achieving_goal
            rewards_achieved = search_node.rewards_achieved
            jump_distances = search_node.jump_distances

            if simulation_goal != -1:
                assert minimum_depth_goal != np.inf
                count_goal += 1
            if minimum_depth_goal != np.inf:
                assert simulation_goal != -1

            list_simulation_goal.append(simulation_goal)
            list_minimum_depth_goal.append(minimum_depth_goal)
            list_steps_to_goals.append(steps_to_goals)
            list_rewards_achieved.append(rewards_achieved)
            list_jump_distances.append(jump_distances)

            my_stdout.close()

    list_num_tokens = np.array(list_num_tokens)
    list_simulation_goal = np.array(list_simulation_goal)
    list_minimum_depth_goal = np.array(list_minimum_depth_goal)
    list_steps_to_goals = np.array(list_steps_to_goals)
    list_rewards_achieved = np.array(list_rewards_achieved)
    list_jump_distances = np.array(list_jump_distances)

    print(f'search_method {search_method}')
    print(f'best_child_method {best_child_method}')
    print(f'count {count}')
    print(f'goal_rate: {count_goal / count:.4f}')
    print(f'num_tokens: {np.mean(list_num_tokens):.4f} +- {np.std(list_num_tokens):.4f}')
    print(f'simulation_goal: {np.mean(list_simulation_goal):.4f} +- {np.std(list_simulation_goal):.4f}')
    print(f'rewards_achieved: {np.mean(list_rewards_achieved):.4f} +- {np.std(list_rewards_achieved):.4f}')
    print(f'jump_distances: {np.mean(list_jump_distances):.4f} +- {np.std(list_jump_distances):.4f}')

    dict_result = {
        'method': method,
        'search_method': search_method,
        'best_child_method': best_child_method,
        'width': width,
        'height': height,
        'wall_density_test': wall_density_test,
        'rewards': rewards,
        'num_simulations_test': num_simulations,
        'max_tree_depth_test': max_tree_depth,
        'seed_base': seed_base,
        'count': count,
        'count_goal': count_goal,
        'goal_rate': count_goal / count,
        'num_tokens': list_num_tokens,
        'simulation_goal': list_simulation_goal,
        'minimum_depth_goal': list_minimum_depth_goal,
        'steps_to_goals': list_steps_to_goals,
        'rewards_achieved': list_rewards_achieved,
        'jump_distances': list_jump_distances,
    }

    path_references = '../results/references'
    os.makedirs(path_references, exist_ok=True)
    str_rewards = '_'.join([f'{reward:.1f}' for reward in rewards])
    str_file = f'navigation_reference_{method}_{width}_{height}_{wall_density_test}_{str_rewards}_{num_simulations}_{max_tree_depth}_{seed_base}.npy'

    np.save(os.path.join(path_references, str_file), dict_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--wall_density_test', type=float, required=True)
    parser.add_argument('--num_simulations', type=int, required=True)
    parser.add_argument('--max_tree_depth', type=int, required=True)
    parser.add_argument('--seed_base', type=int, required=True)

    args = parser.parse_args()

    method = args.method
    width = args.width
    height = args.height
    wall_density_test = args.wall_density_test
    num_simulations = args.num_simulations
    max_tree_depth = args.max_tree_depth
    seed_base = args.seed_base

    num_goals = 3
    rewards = get_rewards(num_goals)

    str_rewards = ', '.join([f'{reward:.1f}' for reward in rewards])

    print('=' * 30)
    print(f'method {method}')
    print(f'width {width}')
    print(f'height {height}')
    print(f'wall_density_test {wall_density_test}')
    print(f'num_goals {num_goals}')
    print(f'rewards {str_rewards}')
    print(f'num_simulations {num_simulations}')
    print(f'max_tree_depth {max_tree_depth}')
    print(f'seed_base {seed_base}')
    print('=' * 30)
    print('')

    get_results(method, width, height, wall_density_test, rewards, num_simulations, max_tree_depth, seed_base)