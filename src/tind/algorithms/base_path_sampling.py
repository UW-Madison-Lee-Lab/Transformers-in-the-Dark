import numpy as np

from tind.algorithms import base_node
from tind.algorithms import utils

from tind import tree_traversal


class BasePathSamplingNode(base_node.BaseNode):
    def __init__(self, state, num_simulations, max_depth_tree, num_rollouts, parent, best_child_method, consider_depth_in_rewards, concatenate_trajectory, random_state, verbose):
        super().__init__(state, parent, num_simulations, max_depth_tree, num_rollouts, consider_depth_in_rewards, concatenate_trajectory, random_state, verbose)

        self.best_child_method = best_child_method
        self.fully_explored = False

        self.param_balancing = 0.1

    def status(self):
        unexplored_children = self.unexplored_children()

        if len(self.children) == 0 and len(unexplored_children) == 0:
            str_status = 'dead_end'
        elif len(self.children) != 0 and len(unexplored_children) == 0:
            str_status = 'fully_expanded'
        else:
            str_status = 'not_fully_expanded'

        return str_status

#    def fully_explored_update(self, node):
#        all_unexplored_children = tree_traversal.get_all_unexplored_children(node)

#        if len(all_unexplored_children) == 0:
#            node.fully_explored = True

#        for child in node.children:
#            if not child.fully_explored:
#                self.fully_explored_update(child)

#        return

    def fully_explored_update(self, node, all_unexplored_children):
        if len(all_unexplored_children) == 0:
            node.fully_explored = True

        for child in node.children:
            if not child.fully_explored:
                new_all_unexplored_children = []

                for unexplored_child in all_unexplored_children:
                    assert node.state.position == unexplored_child[0].position

                    if unexplored_child[1] != 'unexplored' and child.state.position == unexplored_child[1].position:
                        new_all_unexplored_children.append(unexplored_child[1:])

                self.fully_explored_update(child, new_all_unexplored_children)

        return

    def backpropagation(self, reward):
        current_node = self

        while current_node is not None:
            current_node.num_visits += 1
            current_node.result += reward

            current_node = current_node.parent

        return

    def iteration(self, index_simulation):
        if self.verbose:
            print('start_of_iteration')

        all_unexplored_children = tree_traversal.get_all_unexplored_children(self)
        str_all_unexplored_children = tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children, show_full_trajectory=True, concatenate_trajectory=self.concatenate_trajectory)

        if self.verbose:
            print(f'{str_all_unexplored_children}')

        if len(all_unexplored_children) > 0:
            v = self.policy_tree()
            root_node, trajectory = self.to_root(v)

            ind_child = tree_traversal.get_trajectory_index(all_unexplored_children, trajectory)

            if self.verbose:
                print(f'selected_child_and_then_reward {ind_child}')

            current_depth = len(trajectory) - 2 - 1 # exclude 'unexplored' and last node, and root node
            reward = self.rollouts(v, current_depth)
            v.backpropagation(reward)

            all_unexplored_children = tree_traversal.get_all_unexplored_children(root_node)
            self.fully_explored_update(root_node, all_unexplored_children)

            str_reward = f'{reward:.2f}'
        else:
            if self.verbose:
                print(f'selected_child_and_then_reward none')

            trajectory = None
            reward = str_reward = 'none'

        if self.verbose:
            print(f'{str_reward}')

        return trajectory, reward
