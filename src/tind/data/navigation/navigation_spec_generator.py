import numpy as np
from typing import Union, List

from tind.data.navigation.grid_pos import GridPos
from tind.data.navigation.navigation_spec import NavigationSpec


def generate_navigation_spec(
    width: int=8,
    height: int=8,
    wall_density: float=0.5,
    rewards: List[float]=[1.0, 1.0],
    random_state: Union[None, int, np.random.RandomState]=None
) -> NavigationSpec:
    assert isinstance(width, int)
    assert isinstance(height, int)
    assert isinstance(wall_density, float)
    assert isinstance(rewards, (list, np.ndarray))

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random_state = random_state
    else:
        raise ValueError

    assert rewards[0] == 1.0
    for ind in range(0, len(rewards) - 1):
        assert rewards[ind] >= rewards[ind + 1]

    start = GridPos(0, 0)
    main_goal = GridPos(width - 1, height - 1)

    total_cells = width * height
    num_walls = int(wall_density * total_cells)

    available_positions = [
        GridPos(x, y)
        for y in range(0, height)
        for x in range(0, width)
        if (x, y) not in [(start.x, start.y), (main_goal.x, main_goal.y)]
    ]

    num_walls = min(num_walls, len(available_positions))
    walls = list(random_state.choice(available_positions, size=num_walls, replace=False))

    free_positions = list(set(available_positions) - set(walls))
    extra_goals_count = len(rewards) - 1
    
    extra_goals = random_state.choice(free_positions, size=extra_goals_count, replace=False)
    goals = [main_goal] + list(extra_goals)

    navigation_spec = NavigationSpec(
        width=width,
        height=height,
        start=start,
        goals=goals,
        rewards=rewards,
        walls=walls
    )

    return navigation_spec