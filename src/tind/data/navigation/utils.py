import numpy as np
import os

from tind.data.navigation.grid_pos import GridPos


def print_navigation(navigation_spec):
    cell_width = 4

    goal_rewards = {g: f"{r:.1f}" for g, r in zip(navigation_spec.goals, navigation_spec.rewards)}

    # Helper to convert a float reward into a short label (e.g., "1.0" -> "1", "2.5" -> "2.5")
    def format_reward(value: float) -> str:
        # Format to 1 decimal place, then strip trailing zero/decimal if possible
        formatted = f"{value:.1f}".rstrip('0').rstrip('.')
        return formatted

    for y in range(navigation_spec.height - 1, -1, -1):
        str_row = ""

        for x in range(0, navigation_spec.width):
            pos = GridPos(x, y)

            if pos == navigation_spec.start:
                cell_label = "S"
            elif pos in navigation_spec.walls:
                cell_label = "#"
            elif pos in goal_rewards:
                cell_label = f"G{goal_rewards[pos]}"
            else:
                cell_label = "."

            str_row += cell_label.ljust(cell_width)

        print(str_row)


def get_file_name(method, num_mazes, width, height, wall_densities, num_traces, num_simulations, max_tree_depth, num_rollouts):
    str_postfix = f'{method}_{num_mazes}_{width}_{height}_{wall_densities}_{num_traces}_{num_simulations}_{max_tree_depth}_{num_rollouts}'

    str_directory = os.path.join('../results/data/navigation', str_postfix)
    str_file = f'input.txt'

    return str_directory, str_file


def get_rewards(num_goals):
    if num_goals == 3:
        rewards = [1.0, 0.5, 0.2]
    else:
        raise ValueError

    rewards = np.array(rewards)
    return rewards