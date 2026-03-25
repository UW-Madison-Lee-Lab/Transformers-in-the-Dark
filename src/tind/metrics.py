import numpy as np


def normalized_path_lengths(steps_to_goals, minimum_depth_goal):
    assert steps_to_goals.ndim == 2
    assert minimum_depth_goal.ndim == 1
    assert steps_to_goals.shape[0] == minimum_depth_goal.shape[0]

    def normalize(groundtruth_path_length, path_length):
        return np.exp(groundtruth_path_length - path_length)

    steps_to_best_goal = steps_to_goals[:, 0]

    assert minimum_depth_goal.shape[0] == steps_to_best_goal.shape[0]
    normalized_path_length = normalize(steps_to_best_goal, minimum_depth_goal)

    indices_success = np.isfinite(minimum_depth_goal)
    normalized_path_length_without_failures = normalize(steps_to_best_goal[indices_success], minimum_depth_goal[indices_success])
    if len(normalized_path_length_without_failures) == 0:
        normalized_path_length_without_failures = np.array([0.0])

    return normalized_path_length, normalized_path_length_without_failures


def maximum_reward_achieved(rewards_achieved):
    assert rewards_achieved.ndim == 2
    return np.max(rewards_achieved, axis=1)


def cumulative_reward_achieved(rewards_achieved, normalization_factor=None):
    assert rewards_achieved.ndim == 2
    cumulative_reward = []

    for reward_achieved in rewards_achieved:
        new_reward_achieved = reward_achieved[reward_achieved != 0.0]

        if len(new_reward_achieved) == 0:
            cumulative_reward.append(0.0)
        else:
            cumulative_reward.append(np.sum(new_reward_achieved))

    cumulative_reward = np.array(cumulative_reward)

    if normalization_factor is not None:
        assert isinstance(normalization_factor, float)
        cumulative_reward /= normalization_factor

    return cumulative_reward


def ndcg_style_simulation_index(simulation_indices, num_simulations_test):
    assert simulation_indices.ndim == 1
    assert num_simulations_test.ndim == 1
    assert simulation_indices.shape[0] == num_simulations_test.shape[0]

    def metric(iteration, budget):
        if iteration == -1 or iteration > budget:
            return 0.0
        return 1.0 / np.log2(iteration + 1)

    return np.array([metric(iteration, budget) for iteration, budget in zip(simulation_indices, num_simulations_test)])


def normalized_num_tokens(nums_tokens, num_simulations, normalization_factor):
    assert nums_tokens.ndim == 1
    assert num_simulations.ndim == 1
    assert nums_tokens.shape[0] == num_simulations.shape[0]

    return nums_tokens / num_simulations / normalization_factor


def normalized_mean_jump_distance(jump_distances, normalization_factor=4.0):
    assert jump_distances.ndim == 2

    return np.mean(jump_distances, axis=1) / normalization_factor


def normalized_std_jump_distance(jump_distances, normalization_factor=3.0):
    assert jump_distances.ndim == 2

    return np.std(jump_distances, axis=1) / normalization_factor