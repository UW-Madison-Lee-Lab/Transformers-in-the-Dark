import numpy as np


def get_random_states_train(seed_base):
    random_state_problem = np.random.RandomState(seed_base)
    random_state_search = np.random.RandomState(seed_base)

    return random_state_problem, random_state_search


def get_random_states_test(seed_base):
    random_state_problem = np.random.RandomState(seed_base + 424242)
    random_state_search = np.random.RandomState(seed_base + 4242)

    return random_state_problem, random_state_search


def ci(vals):
    return 1.96 * np.std(vals, ddof=1) / np.sqrt(vals.shape[0])