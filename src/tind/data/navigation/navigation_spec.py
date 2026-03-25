# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
)

from tind.data.navigation.grid_pos import GridPos


@dataclass
class NavigationSpec:
    width: int
    height: int
    start: GridPos
    goals: Sequence[GridPos]
    rewards: Sequence[float]
    walls: Sequence[GridPos]
    num_all_actions = 4

    def is_pos_in_spec(self, pos: GridPos) -> bool:
        x_pos_range = pos.x >= 0 and pos.x < self.width
        y_pos_range = pos.y >= 0 and pos.y < self.height

        is_wall = False

        for wall in self.walls:
            is_wall = is_wall or (wall == pos)

        return x_pos_range and y_pos_range and not is_wall

    def is_pos_in_wall(self, pos: GridPos) -> bool:
        for wall in self.walls:
            if pos == wall:
                return True
        return False

    def neighborhood(self, pos: GridPos) -> Sequence[GridPos]:
        neigh: List[GridPos] = []

        up_pos = pos.up
        if self.is_pos_in_spec(up_pos):
            neigh.append(up_pos)

        down_pos = pos.down
        if self.is_pos_in_spec(down_pos):
            neigh.append(down_pos)

        left_pos = pos.left
        if self.is_pos_in_spec(left_pos):
            neigh.append(left_pos)

        right_pos = pos.right
        if self.is_pos_in_spec(right_pos):
            neigh.append(right_pos)

        assert len(neigh) <= self.num_all_actions
        return neigh

    @property
    def boundary_walls(self) -> Sequence[GridPos]:
        w, h = self.width, self.height

        top = [GridPos(x, h) for x in range(-1, w + 1)]
        bottom = [GridPos(x, -1) for x in range(-1, w + 1)]
        left = [GridPos(-1, y) for y in range(-1, h + 1)]
        right = [GridPos(w, y) for y in range(-1, h + 1)]

        return top + bottom + left + right