import numpy as np
import os
import io
import sys
import argparse

from tind.utils import get_random_states_test

from tind.algorithms.selection import select_method
from tind.algorithms.utils import get_search_method_best_child_method

from tind.data.tree_search.generate import get_random_tree_search
from tind.data.tree_search.utils import get_rewards


def get_results(method, num_arms, max_depth, rewards, num_simulations, max_tree_depth, seed_base):
    assert num_arms > 0
    assert max_depth > 0
    assert num_simulations > 0
    assert max_tree_depth > 0

    search_method, best_child_method = get_search_method_best_child_method(method)
    random_state_problem, random_state_search = get_random_states_test(seed_base)

    num_instances_test = 10
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

    for ind_instance in range(0, num_instances_test):
        start_state = get_random_tree_search(num_arms, max_depth, rewards, random_state_problem)
        steps_to_goals = [len(goal) - 1 for goal in start_state.tree_search_spec.goals]

        for ind_trace in range(0, num_traces):
            old_stdout = sys.stdout
            my_stdout = sys.stdout = io.StringIO()

            count += 1

            search_node = select_method(
                method, start_state, None, num_simulations, max_tree_depth, num_rollouts, True, random_state_search, True)
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
        'num_arms': num_arms,
        'max_depth': max_depth,
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
    str_file = f'tree_search_reference_{method}_{num_arms}_{max_depth}_{str_rewards}_{num_simulations}_{max_tree_depth}_{seed_base}.npy'

    np.save(os.path.join(path_references, str_file), dict_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--num_goals', type=int, required=True)
    parser.add_argument('--num_arms', type=int, required=True)
    parser.add_argument('--max_depth', type=int, required=True)
    parser.add_argument('--num_simulations', type=int, required=True)
    parser.add_argument('--max_tree_depth', type=int, required=True)
    parser.add_argument('--seed_base', type=int, required=True)

    args = parser.parse_args()

    method = args.method
    num_goals = args.num_goals
    num_arms = args.num_arms
    max_depth = args.max_depth
    num_simulations = args.num_simulations
    max_tree_depth = args.max_tree_depth
    seed_base = args.seed_base

    rewards = get_rewards(num_goals)

    str_rewards = ', '.join([f'{reward:.1f}' for reward in rewards])

    print('=' * 30)
    print(f'method {method}')
    print(f'num_arms {num_arms}')
    print(f'max_depth {max_depth}')
    print(f'num_goals {num_goals}')
    print(f'rewards {str_rewards}')
    print(f'num_simulations {num_simulations}')
    print(f'max_tree_depth {max_tree_depth}')
    print(f'seed_base {seed_base}')
    print('=' * 30)
    print('')

    get_results(method, num_arms, max_depth, rewards, num_simulations, max_tree_depth, seed_base)