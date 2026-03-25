import numpy as np

from tind.algorithms import base_node
from tind.algorithms import utils

from tind import tree_traversal


class GreedyLeafSamplingNode(base_node.BaseNode):
    def __init__(self, state, num_simulations, max_tree_depth, num_rollouts, parent=None, consider_depth_in_rewards=False, concatenate_trajectory=False, random_state=None, verbose=False):
        super().__init__(state, parent, num_simulations, max_tree_depth, num_rollouts, consider_depth_in_rewards, concatenate_trajectory, random_state, verbose)

    def policy_tree(self, all_unexplored_children):
        values = []

        for child in all_unexplored_children:
            parent_node = self.traversal(child)
            values.append(parent_node.q())

        index_best_child = utils.argmax_random_tie_breaking(
            values,
            random_state=self.random_state
        )
        best_child_chosen = all_unexplored_children[index_best_child]

        parent_node = self.traversal(best_child_chosen)
        next_state = best_child_chosen[-1]
        child_node = GreedyLeafSamplingNode(
            next_state,
            self.num_simulations,
            self.max_tree_depth,
            self.num_rollouts,
            parent=parent_node,
            consider_depth_in_rewards=self.consider_depth_in_rewards,
            concatenate_trajectory=self.concatenate_trajectory,
            random_state=self.random_state,
            verbose=self.verbose
        )
        parent_node.children.append(child_node)

        return child_node

    def backpropagation(self, reward):
        self.num_visits += 1
        self.result += reward

    def iteration(self, index_simulation):
        if self.verbose:
            print('start_of_iteration')

        all_unexplored_children = tree_traversal.get_all_unexplored_children(self)
        str_all_unexplored_children = tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children, show_full_trajectory=True, concatenate_trajectory=self.concatenate_trajectory)

        if self.verbose:
            print(f'{str_all_unexplored_children}')

        if len(all_unexplored_children) > 0:
            v = self.policy_tree(all_unexplored_children)
            _, trajectory = self.to_root(v)

            ind_child = tree_traversal.get_trajectory_index(all_unexplored_children, trajectory)

            if self.verbose:
                print(f'selected_child_and_then_reward {ind_child}')

            current_depth = len(trajectory) - 2 - 1 # exclude 'unexplored' and last node, and root node
            reward = self.rollouts(v, current_depth)
            v.backpropagation(reward)

            str_reward = f'{reward:.2f}'
        else:
            if self.verbose:
                print(f'selected_child_and_then_reward none')

            trajectory = None
            reward = str_reward = 'none'

        if self.verbose:
            print(f'{str_reward}')

        return trajectory, reward
