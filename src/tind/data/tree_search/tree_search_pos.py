from dataclasses import dataclass
from typing import Any, Dict, Union


@dataclass
class TreeSearchPos:
    arm_index: Union[None, int]
    depth: int

    def __str__(self) -> str:
        if self.arm_index is not None:
            return f'i{self.arm_index}d{self.depth}'
        else:
            return f'r0d{self.depth}'

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, TreeSearchPos)
        return self.arm_index == other.arm_index and self.depth == other.depth