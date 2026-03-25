import copy


def get_all_unexplored_children(mcts_node):
    all_unexplored_children = []

    def traverse(trajectory, node):
        trajectory.append(node.state)

        unexplored_children = node.state.get_unexplored_children(node.children, node.parent)
        for child in unexplored_children:
            all_unexplored_children.append(trajectory + ['unexplored', child])

        for child in node.children:
            traverse(copy.deepcopy(trajectory), child)

    traverse([], mcts_node)
    return all_unexplored_children


def convert_unexplored_child_to_string(trajectory, show_full_trajectory=False, index_simulation=None, concatenate_trajectory=False):
    assert trajectory[-2] == 'unexplored'

    if concatenate_trajectory:
        separator = '>'
    else:
        separator = ' '

    if show_full_trajectory:
        str_child = separator.join([f'{str(child.position)}' for child in trajectory[:-2]])
        str_child += separator
        str_child += f'{str(trajectory[-1].position)}'
    else:
        str_child = f's{trajectory[-3].index_simulation} {str(trajectory[-3].position)}'
        str_child += ' '
        if index_simulation is not None:
            str_child += f's{index_simulation + 1} {str(trajectory[-1].position)}'
        else:
            str_child += f'* {str(trajectory[-1].position)}'

    return str_child


def convert_all_unexplored_children_to_string_list(trajectories, show_full_trajectory=False, concatenate_trajectory=False):
    list_str_child = []

    for trajectory in trajectories:
        list_str_child.append(convert_unexplored_child_to_string(trajectory, show_full_trajectory=show_full_trajectory, concatenate_trajectory=concatenate_trajectory))

    return list_str_child


def convert_all_unexplored_children_to_strings(trajectories, show_full_trajectory=False, concatenate_trajectory=False):
    list_str_child = convert_all_unexplored_children_to_string_list(trajectories, show_full_trajectory=show_full_trajectory, concatenate_trajectory=concatenate_trajectory)

    str_children = ' '.join([f'{ind_child} {str_child}' for ind_child, str_child in enumerate(list_str_child)])
#    str_children = ' / '.join([f'{ind_child} {str_child}' for ind_child, str_child in enumerate(list_str_child)])
#    str_children = ' / '.join([f'{str_child}' for str_child in list_str_child])

    if str_children == '':
        str_children = 'none'

    return str_children


def get_trajectory_index_no_assertion(unexplored_trajectories, trajectory):
    str_trajectory = convert_unexplored_child_to_string(trajectory, show_full_trajectory=True)
    trajectory_index = -1

    for ind_trajectory, unexplored_trajectory in enumerate(unexplored_trajectories):
        str_unexplored_trajectory = convert_unexplored_child_to_string(unexplored_trajectory, show_full_trajectory=True)

        if str_trajectory == str_unexplored_trajectory:
            trajectory_index = ind_trajectory
            break

    return trajectory_index


def get_trajectory_index(unexplored_trajectories, trajectory):
    trajectory_index = get_trajectory_index_no_assertion(unexplored_trajectories, trajectory)

    assert trajectory_index != -1
    return trajectory_index