import numpy as np


def argmax_random_tie_breaking(arr, random_state=None):
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random_state = random_state
    else:
        raise ValueError

    return random_state.choice(np.where(arr == np.max(arr))[0])


def get_search_method_best_child_method(method):
    if 'pps_' in method:
        list_method = method.split('_')

        search_method = list_method[0]
        best_child_method = '_'.join(list_method[1:])
    else:
        search_method = method
        best_child_method = None

    return search_method, best_child_method


if __name__ == '__main__':
    arr = np.array([4, 3, 9, 1, 5, 3, 9, 8, 3, 8, 9, 2])

    print(argmax_random_tie_breaking(arr, random_state=None))
