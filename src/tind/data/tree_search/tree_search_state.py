import numpy as np
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
)

from tind.data.tree_search.tree_search_pos import TreeSearchPos
from tind.data.tree_search.tree_search_spec import TreeSearchSpec


class TreeSearchState:
    def __init__(
        self,
        tree_search_spec: TreeSearchSpec,
        index_simulation: int,
        path: Union[None, Sequence["TreeSearchPos"]],
    ):
        assert np.all(0 <= tree_search_spec.rewards)
        assert np.all(tree_search_spec.rewards <= 1)
        if path is None:
            path = [TreeSearchPos(arm_index=None, depth=0)]

        if len(path) == 1:
            assert path[-1].arm_index == None
        else:
            assert 0 <= path[-1].arm_index < tree_search_spec.num_arms
        assert 0 <= path[-1].depth <= tree_search_spec.max_depth

        self.tree_search_spec = tree_search_spec
        self.index_simulation = index_simulation
        self.position = path[-1]
        self.path = path

    @property
    def state(self) -> Dict[str, Any]:
        return {"arm_index": self.position.arm_index, "depth": self.position.depth}

    def get_children(self) -> List["TreeSearchState"]:
        children = []

        if self.position.depth < self.tree_search_spec.max_depth:
            for ind_child in range(0, self.tree_search_spec.num_arms):
                children.append(
                    TreeSearchState(self.tree_search_spec, 0, self.path + [TreeSearchPos(ind_child, self.position.depth + 1)])
                )

        assert len(children) <= self.tree_search_spec.num_arms
        return children

    def get_children_satisfying_constraints(
        self,
        parent: "BaseNode"
    ) -> List["TreeSearchState"]:
        # parent is not used.

        children = self.get_children()
        children_satisfying_constraints = children

        assert len(children_satisfying_constraints) <= self.tree_search_spec.num_arms
        return children_satisfying_constraints

    def get_unexplored_children(
        self,
        explored_children: List["BaseNode"],
        parent: "BaseNode"
    ) -> List["TreeSearchState"]:
        children = self.get_children_satisfying_constraints(parent)
        unexplored_children = []

        for child_state in children:
            is_explored = False

            for explored_child in explored_children:
                is_explored = is_explored or (child_state.position == explored_child.state.position)

            if not is_explored:
                unexplored_children.append(child_state)

        assert len(unexplored_children) <= self.tree_search_spec.num_arms
        return unexplored_children

    @property
    def goal_index(self) -> int:
        index = -1

        if self.position.depth == self.tree_search_spec.max_depth:
            for ind_goal, goal in enumerate(self.tree_search_spec.goals):
                is_same = True

                for elem_path, elem_goal in zip(self.path, goal):
                    if elem_path == elem_goal:
                        pass
                    else:
                        is_same = False
                        break

                if is_same:
                    index = ind_goal
                    break

        return index

    @property
    def is_goal(self) -> bool:
        index = self.goal_index
        return index != -1

    @property
    def result(self) -> float:
        assert len(self.tree_search_spec.goals) == len(self.tree_search_spec.rewards)

        index = self.goal_index

        if index == -1:
            return 0.0
        else:
            return self.tree_search_spec.rewards[index]