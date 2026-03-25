from typing import (
    Any,
    Dict,
    List,
    Union,
)


class NavigationState:
    def __init__(
        self,
        navigation_spec: "NavigationSpec",
        index_simulation: int,
        position: Union[None, "GridPos"],
    ):
        self.navigation_spec = navigation_spec
        self.index_simulation = index_simulation

        if position is None:
            self.position = navigation_spec.start
        else:
            self.position = position

    @property
    def state(self) -> Dict[str, Any]:
        return {"x": self.position.x, "y": self.position.y}

    def get_children(self) -> List["NavigationState"]:
        children = []

        for child_pos in self.navigation_spec.neighborhood(self.position):
            children.append(
                NavigationState(
                    self.navigation_spec,
                    0,
                    child_pos,
                )
            )

        assert len(children) <= self.navigation_spec.num_all_actions
        return children

    def get_children_satisfying_constraints(
        self,
        parent: "BaseNode"
    ) -> List["NavigationState"]:
        children = self.get_children()
        children_satisfying_constraints = []

        allow_go_back = True

        for child_state in children:
            if parent is not None:
                if not allow_go_back:
                    satisfy_constraints = not (child_state.position == parent.state.position)
                else:
                    satisfy_constraints = True
            else:
                satisfy_constraints = True

            if satisfy_constraints:
                children_satisfying_constraints.append(child_state)

        for goal in self.navigation_spec.goals:
            if self.position == goal:
                children_satisfying_constraints = []
                break

        assert len(children_satisfying_constraints) <= self.navigation_spec.num_all_actions
        return children_satisfying_constraints

    def get_unexplored_children(
        self,
        explored_children: List["BaseNode"],
        parent: "BaseNode"
    ) -> List["NavigationState"]:
        children = self.get_children_satisfying_constraints(parent)
        unexplored_children = []

        for child_state in children:
            is_explored = False

            for explored_child in explored_children:
                is_explored = is_explored or (child_state.position == explored_child.state.position)

            if not is_explored:
                unexplored_children.append(child_state)

        assert len(unexplored_children) <= self.navigation_spec.num_all_actions
        return unexplored_children

    @property
    def goal_index(self) -> int:
        index = -1

        for ind_goal, goal in enumerate(self.navigation_spec.goals):
            if self.position == goal:
                index = ind_goal
                break

        return index

    @property
    def is_goal(self) -> bool:
        index = self.goal_index
        return index != -1

    @property
    def result(self) -> float:
        assert len(self.navigation_spec.goals) == len(self.navigation_spec.rewards)

        index = self.goal_index

        if index == -1:
            return 0.0
        else:
            return self.navigation_spec.rewards[index]