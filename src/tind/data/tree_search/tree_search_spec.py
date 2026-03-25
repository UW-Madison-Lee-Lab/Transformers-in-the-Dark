from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Sequence,
    Tuple,
)


@dataclass
class TreeSearchSpec:
    num_arms: int
    goals: Sequence[Sequence["TreeSearchPos"]]
    rewards: Sequence[float]
    max_depth: int