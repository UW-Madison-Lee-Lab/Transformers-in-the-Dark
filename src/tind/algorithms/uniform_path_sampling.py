import numpy as np

from tind.algorithms import base_path_sampling
from tind.algorithms import utils

from tind import tree_traversal


class UniformPathSamplingNode(base_path_sampling.BasePathSamplingNode):
    def __init__(self, state, num_simulations, max_tree_depth, num_rollouts, parent=None, consider_depth_in_rewards=False, concatenate_trajectory=False, random_state=None, verbose=False):
        best_child_method = 'random' # it is not used.

        super().__init__(state, num_simulations, max_tree_depth, num_rollouts, parent, best_child_method, consider_depth_in_rewards, concatenate_trajectory, random_state, verbose)

    def best_child(self):
        len_children = len(self.children)
        unexplored_children = self.unexplored_children()

        values = [self.random_state.uniform(low=0, high=1) for child in self.children]
        values = [values[ind_child] if not child.fully_explored else -np.inf for ind_child, child in enumerate(self.children)]
        values += [self.random_state.uniform(low=0, high=1) for _ in range(0, len(unexplored_children))]
        assert len(values) == len(self.state.get_children_satisfying_constraints(self.parent))

        index_best_child = utils.argmax_random_tie_breaking(
            values,
            random_state=self.random_state
        )

        if index_best_child < len_children:
            add_new_child = False
            best_child_chosen = self.children[index_best_child]
        else:
            add_new_child = True
            best_state_chosen = unexplored_children[index_best_child - len_children]
            best_child_chosen = UniformPathSamplingNode(
                best_state_chosen,
                self.num_simulations,
                self.max_tree_depth,
                self.num_rollouts,
                parent=self,
                consider_depth_in_rewards=self.consider_depth_in_rewards,
                concatenate_trajectory=self.concatenate_trajectory,
                random_state=self.random_state,
                verbose=self.verbose
            )

            self.children.append(best_child_chosen)

        return best_child_chosen, add_new_child

    def policy_tree(self):
        current_node = self

        while not current_node.state.is_goal:
            str_status = current_node.status()

            if str_status == 'fully_expanded' or str_status == 'not_fully_expanded':
                current_node, add_new_child = current_node.best_child()

                if add_new_child:
                    return current_node
            elif str_status == 'dead_end':
                return current_node
            else:
                raise ValueError

        return current_node
