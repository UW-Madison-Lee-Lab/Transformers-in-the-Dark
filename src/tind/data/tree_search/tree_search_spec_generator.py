import numpy as np
from typing import Union, List

from .tree_search_spec import TreeSearchSpec
from .utils import get_path


def sample_path(num_arms, max_depth, random_state):
    path = random_state.choice(num_arms, size=max_depth, replace=True)
    path = get_path(path)

    return path


def generate_tree_search_spec(
    num_arms: int=2,
    max_depth: int=4,
    rewards: List[float]=[1.0, 1.0],
    random_state: Union[None, int, np.random.RandomState]=None
) -> TreeSearchSpec:
    assert isinstance(num_arms, int)
    assert isinstance(rewards, (list, np.ndarray))

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random_state = random_state
    else:
        raise ValueError

    goals = []
    for _ in range(0, len(rewards)):
        goals.append(sample_path(num_arms, max_depth, random_state))

    tree_search_spec = TreeSearchSpec(
        num_arms=num_arms,
        goals=goals,
        rewards=rewards,
        max_depth=max_depth,
    )
    return tree_search_spec