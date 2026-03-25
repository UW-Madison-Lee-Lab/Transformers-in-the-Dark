import numpy as np

from tind.data.navigation.grid_pos import GridPos
from tind.data.navigation.navigation_spec import NavigationSpec
from tind.data.navigation.navigation_state import NavigationState

from tind.algorithms.policy_path_sampling import PolicyPathSamplingNode

import tree_traversal as tree_traversal
import tind.data.navigation.utils as utils_navigation


if __name__ == '__main__':
    seed = 42

    print('')
    print('=' * 30)
    print('Navigation')
    print('=' * 30)
    print('')

    width = 4
    height = 4
    start = GridPos(0, 0)
    goals = [GridPos(3, 3)]
    rewards = [1.0]
    walls = [
        GridPos(1, 2),
        GridPos(1, 3),
        GridPos(2, 1),
    ]

    navigation_spec = NavigationSpec(width, height, start, goals, rewards, walls)
    start_state = NavigationState(navigation_spec, 0, None)

    utils_navigation.print_navigation(navigation_spec)

    search_node = PolicyPathSamplingNode(start_state, 5, 10, 10, parent=None, random_state=seed, verbose=True)
    next_action = search_node.next_action()

    all_unexplored_children = tree_traversal.get_all_unexplored_children(search_node)

    print('all_unexplored_children')
    for child in all_unexplored_children:
        print(child)
    print('')

    print(tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children))
    print(tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children, show_full_trajectory=True))
    print(tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children, show_full_trajectory=True, concatenate_trajectory=True))