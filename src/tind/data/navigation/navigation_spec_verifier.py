from collections import deque
from typing import Optional, List
import copy


def is_navigation_spec_solvable(navigation_spec: "NavigationSpec") -> bool:
    def can_reach_specific_goal(goal: "GridPos") -> bool:
        queue = deque([navigation_spec.start])
        visited = set([navigation_spec.start])

        blocked_positions = set(navigation_spec.walls) | {g for g in navigation_spec.goals if g != goal}

        while queue:
            current = queue.popleft()

            if current == goal:
                return True

            for neighbor in navigation_spec.neighborhood(current):
                if neighbor not in blocked_positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    for g in navigation_spec.goals:
        if not can_reach_specific_goal(g):
            return False

    return True


def count_steps_to_goal(navigation_spec: "NavigationSpec", goal: "GridPos") -> Optional[int]:
    start = navigation_spec.start

    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)

    blocked_positions = set(navigation_spec.walls) | {g for g in navigation_spec.goals if g != goal}

    while queue:
        current, dist = queue.popleft()

        if current == goal:
            return dist

        for neighbor in navigation_spec.neighborhood(current):
            if neighbor not in blocked_positions and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    # It won't happen because we assume that navigation_spec is always solvable.
    return None


def count_steps_to_goals(navigation_spec: "NavigationSpec") -> List[int]:
    assert is_navigation_spec_solvable(navigation_spec)

    steps_to_goals = []

    for g in navigation_spec.goals:
        steps_to_goal = count_steps_to_goal(navigation_spec, g)
        steps_to_goals.append(steps_to_goal)

    return steps_to_goals