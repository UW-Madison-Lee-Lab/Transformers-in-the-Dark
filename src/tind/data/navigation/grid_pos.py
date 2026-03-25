# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import Dict


@dataclass
class GridPos:
    x: int
    y: int

    def __hash__(self) -> int:
        assert self.x < 100000
        return int(self.x + 100000 * self.y)

    @property
    def up(self) -> "GridPos":
        return GridPos(self.x, self.y + 1)

    @property
    def down(self) -> "GridPos":
        return GridPos(self.x, self.y - 1)

    @property
    def left(self) -> "GridPos":
        return GridPos(self.x - 1, self.y)

    @property
    def right(self) -> "GridPos":
        return GridPos(self.x + 1, self.y)

    def __str__(self) -> str:
        return f'x{self.x}y{self.y}'

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, GridPos)
        return self.x == other.x and self.y == other.y

    def distance(self, other: "GridPos") -> int:
        x_dist = abs(self.x - other.x)
        y_dist = abs(self.y - other.y)
        return x_dist + y_dist