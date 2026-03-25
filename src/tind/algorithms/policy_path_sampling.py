import numpy as np

from tind.algorithms import base_path_sampling
from tind.algorithms import utils

from tind import tree_traversal


class PolicyPathSamplingNode(base_path_sampling.BasePathSamplingNode):
    def __init__(self, state, num_simulations, max_tree_depth, num_rollouts, parent=None, best_child_method='uct', consider_depth_in_rewards=False, concatenate_trajectory=False, random_state=None, verbose=False):
        super().__init__(state, num_simulations, max_tree_depth, num_rollouts, parent, best_child_method, consider_depth_in_rewards, concatenate_trajectory, random_state, verbose)

    def expansion(self):
        unexplored_children = self.unexplored_children()
        next_state = self.random_state.choice(unexplored_children)
        child_node = PolicyPathSamplingNode(
            next_state,
            self.num_simulations,
            self.max_tree_depth,
            self.num_rollouts,
            parent=self,
            best_child_method=self.best_child_method,
            consider_depth_in_rewards=self.consider_depth_in_rewards,
            concatenate_trajectory=self.concatenate_trajectory,
            random_state=self.random_state,
            verbose=self.verbose
        )

        self.children.append(child_node)
        return child_node

    def best_child(self):
        if self.best_child_method == 'uct':
            values = [
                child.q() / child.n() + self.param_balancing * np.sqrt(2 * np.log(self.n()) / child.n()) for child in self.children
            ]
        elif self.best_child_method == 'greedy':
            values = [child.q() / child.n() for child in self.children]
        elif self.best_child_method == 'random':
            values = [self.random_state.uniform(low=0, high=1) for child in self.children]
        elif self.best_child_method == 'pure_exploration':
            values = [self.n() / child.n() for child in self.children]
        elif self.best_child_method == 'epsilon_greedy':
            epsilon = 0.25

            if self.random_state.uniform(low=0, high=1) < epsilon:
                values = [self.random_state.uniform(low=0, high=1) for child in self.children]
            else:
                values = [child.q() / child.n() for child in self.children]
        else:
            raise ValueError

        values = [values[ind_child] if not child.fully_explored else -np.inf for ind_child, child in enumerate(self.children)]
        index_best_child = utils.argmax_random_tie_breaking(
            values,
            random_state=self.random_state
        )

        best_child_chosen = self.children[index_best_child]

        return best_child_chosen

    def policy_tree(self):
        current_node = self

        while not current_node.state.is_goal:
            str_status = current_node.status()

            if str_status == 'fully_expanded':
                current_node = current_node.best_child()
            elif str_status == 'not_fully_expanded':
                return current_node.expansion()
            elif str_status == 'dead_end':
                return current_node
            else:
                raise ValueError

        return current_node
