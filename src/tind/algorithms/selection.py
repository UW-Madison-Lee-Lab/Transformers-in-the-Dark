from tind.algorithms.uniform_leaf_sampling import UniformLeafSamplingNode
from tind.algorithms.uniform_path_sampling import UniformPathSamplingNode
from tind.algorithms.greedy_leaf_sampling import GreedyLeafSamplingNode
from tind.algorithms.policy_path_sampling import PolicyPathSamplingNode

from tind.algorithms.utils import get_search_method_best_child_method


def select_method(
    method,
    state,
    parent,
    num_simulations,
    max_tree_depth,
    num_rollouts,
    concatenate_trajectory,
    random_state,
    verbose,
):
    consider_depth_in_rewards = False
    search_method, best_child_method = get_search_method_best_child_method(method)

    if search_method == 'pps':
        assert best_child_method is not None

        search_node = PolicyPathSamplingNode(state, num_simulations, max_tree_depth, num_rollouts,
            parent=parent,
            best_child_method=best_child_method,
            consider_depth_in_rewards=consider_depth_in_rewards,
            concatenate_trajectory=concatenate_trajectory,
            random_state=random_state,
            verbose=verbose
        )
    elif search_method == 'uls':
        assert best_child_method is None

        search_node = UniformLeafSamplingNode(state, num_simulations, max_tree_depth, num_rollouts,
            parent=parent,
            consider_depth_in_rewards=consider_depth_in_rewards,
            concatenate_trajectory=concatenate_trajectory,
            random_state=random_state,
            verbose=verbose
        )
    elif search_method == 'ups':
        assert best_child_method is None

        search_node = UniformPathSamplingNode(state, num_simulations, max_tree_depth, num_rollouts,
            parent=parent,
            consider_depth_in_rewards=consider_depth_in_rewards,
            concatenate_trajectory=concatenate_trajectory,
            random_state=random_state,
            verbose=verbose
        )
    elif search_method == 'gls':
        assert best_child_method is None

        search_node = GreedyLeafSamplingNode(state, num_simulations, max_tree_depth, num_rollouts,
            parent=parent,
            consider_depth_in_rewards=consider_depth_in_rewards,
            concatenate_trajectory=concatenate_trajectory,
            random_state=random_state,
            verbose=verbose
        )
    else:
        raise ValueError

    return search_node