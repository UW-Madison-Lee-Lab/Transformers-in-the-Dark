import numpy as np

from tind import tree_traversal
from tind.algorithms import utils
from tind.data.tree_search.tree_search_state import TreeSearchState
from tind.data.navigation.navigation_state import NavigationState


class BaseNode:
    def __init__(
        self,
        state,
        parent,
        num_simulations,
        max_tree_depth,
        num_rollouts,
        consider_depth_in_rewards,
        concatenate_trajectory,
        random_state,
        verbose
    ):
        assert isinstance(consider_depth_in_rewards, bool)
        assert isinstance(concatenate_trajectory, bool)
        assert isinstance(verbose, bool)

        self.state = state
        self.parent = parent
        self.verbose = verbose

        if random_state is None or isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            raise ValueError

        self.children = []
        self.num_visits = 0
        self.result = 0.0

        self.num_simulations = num_simulations
        self.max_tree_depth = max_tree_depth
        self.num_rollouts = num_rollouts
        self.consider_depth_in_rewards = consider_depth_in_rewards

        self.simulation_achieving_goal = -1
        self.minimum_depth_achieving_goal = np.inf
        self.rewards_achieved = []

        self.trajectories = []
        self.distances = []
        self.jump_distances = []

        self.all_unexplored_children = []
        self.concatenate_trajectory = concatenate_trajectory

    def to_root(self, node):
        root_node = node
        trajectory = ['unexplored', root_node.state]

        while root_node.parent is not None:
            root_node = root_node.parent
            trajectory = [root_node.state] + trajectory

        return root_node, trajectory

    def traversal_once(self, current_node, target_child):
        find_child = False

        for node_child in current_node.children:
            if target_child.position == node_child.state.position:
                next_node = node_child
                find_child = True
                break

        if not find_child:
            raise ValueError

        return next_node

    def traversal(self, trajectory):
        current_node, _ = self.to_root(self)

        assert current_node.state.position == trajectory[0].position
        assert trajectory[-2] == 'unexplored'

        for child in trajectory[1:-2]:
            current_node = self.traversal_once(current_node, child)

        return current_node

    def unexplored_children(self):
        unexplored_children = self.state.get_unexplored_children(self.children, self.parent)
        return unexplored_children

    def q(self):
        return self.result

    def n(self):
        return self.num_visits

    def rollout(self, current_depth):
        current_state = self.state

        reward = current_state.result
        depth = 1

        while (current_depth + depth) < self.max_tree_depth and not current_state.is_goal:
            possible_states = current_state.get_children_satisfying_constraints(self.parent)

            if len(possible_states) > 0:
                current_state = self.policy_rollout(possible_states)

                reward += current_state.result
                depth += 1
            else:
                break

        return reward, depth

    def policy_rollout(self, possible_states):
        return self.random_state.choice(possible_states)

    def rollouts(self, node, current_depth):
        # current_depth does not include node.
        rewards = []

        for _ in range(0, self.num_rollouts):
            reward, depth = node.rollout(current_depth)
            if self.consider_depth_in_rewards:
                reward /= depth

            rewards.append(reward)
        reward = np.mean(rewards)

        return reward

    def all_unexplored_children_update(self, all_unexplored_children):
        new_all_unexplored_children = []

        list_str_all_unexplored_children = tree_traversal.convert_all_unexplored_children_to_string_list(all_unexplored_children, show_full_trajectory=True)
        list_str_all_unexplored_children_in_object = tree_traversal.convert_all_unexplored_children_to_string_list(self.all_unexplored_children, show_full_trajectory=True)

        for ind_child, str_unexplored_child in enumerate(list_str_all_unexplored_children):
            if str_unexplored_child not in list_str_all_unexplored_children_in_object:
                new_all_unexplored_children.append(all_unexplored_children[ind_child])

        self.all_unexplored_children += new_all_unexplored_children

        if len(new_all_unexplored_children) > 0:
            str_new_all_unexplored_children = tree_traversal.convert_all_unexplored_children_to_strings(new_all_unexplored_children)
        else:
            if len(self.all_unexplored_children) > 0:
                str_new_all_unexplored_children = 'no_new_children'
            else:
                str_new_all_unexplored_children = 'none'

        return str_new_all_unexplored_children

    def metric_updates(self, ind_simulation, trajectory):
        satisfy_best_goal = False
        reward_achieved = 0.0

        self.trajectories.append(trajectory)
        if len(self.trajectories) > 1:
            new_trajectory_previous = self.trajectories[-2][:-2] + self.trajectories[-2][-1:]
            new_trajectory_current = self.trajectories[-1][:-2] + self.trajectories[-1][-1:]
            find_branch = False

            for ind_previous, elem_previous in enumerate(new_trajectory_previous[::-1]):
                for ind_current, elem_current in enumerate(new_trajectory_current[::-1]):
                    if elem_previous.position == elem_current.position:
                        self.distances.append(ind_previous + ind_current)
                        find_branch = True

                    if find_branch:
                        break
                if find_branch:
                    break

        if len(self.distances) > 1:
            self.jump_distances.append((self.distances[-2] + self.distances[-1]) / 2)

        if isinstance(self.state, NavigationState):
            if self.state.navigation_spec.goals[0] == trajectory[-1].position:
                satisfy_best_goal = True

            for ind_goal, goal in enumerate(self.state.navigation_spec.goals):
                if goal == trajectory[-1].position:
                    reward_achieved = self.state.navigation_spec.rewards[ind_goal]
                    break
        elif isinstance(self.state, TreeSearchState):
            assert trajectory[-2] == 'unexplored'
            new_trajectory = trajectory[:-2] + trajectory[-1:]

            if len(new_trajectory) == (self.state.tree_search_spec.max_depth + 1):
                assert (self.state.tree_search_spec.max_depth + 1) == len(self.state.tree_search_spec.goals[0])
                satisfy_best_goal = True

                for elem_trajectory, elem_goal in zip(new_trajectory, self.state.tree_search_spec.goals[0]):
                    satisfy_best_goal = satisfy_best_goal and (elem_trajectory.position == elem_goal)

                for ind_goal, goal in enumerate(self.state.tree_search_spec.goals):
                    satisfy_goal = True

                    for elem_trajectory, elem_goal in zip(new_trajectory, goal):
                        satisfy_goal = satisfy_goal and (elem_trajectory.position == elem_goal)

                    if satisfy_goal:
                        reward_achieved = self.state.tree_search_spec.rewards[ind_goal]
                        break
        else:
            raise ValueError

        if satisfy_best_goal:
            self.minimum_depth_achieving_goal = np.minimum(self.minimum_depth_achieving_goal, len(trajectory) - 1 - 1) # remove 'unexplored' and the root node

            if self.simulation_achieving_goal == -1:
                self.simulation_achieving_goal = ind_simulation + 1

        self.rewards_achieved.append(reward_achieved)

    def iteration(self, index_simulation):
        pass

    def next_action(self):
        if self.verbose:
            print('start_of_round')

        for ind_simulation in range(0, self.num_simulations):
            trajectory, _ = self.iteration(ind_simulation)

            if trajectory is not None:
                leaf_node = self.traversal(trajectory)
                leaf_node = self.traversal_once(leaf_node, trajectory[-1])
                leaf_node.state.index_simulation = ind_simulation + 1

                self.metric_updates(ind_simulation, trajectory)

        values = [child.q() / child.n() if child.n() > 0 else -np.inf for child in self.children]
        index_next_action = utils.argmax_random_tie_breaking(
            values,
            random_state=self.random_state
        )
        next_action_chosen = self.children[index_next_action]

        if self.verbose:
            print('end_of_round')

        return next_action_chosen