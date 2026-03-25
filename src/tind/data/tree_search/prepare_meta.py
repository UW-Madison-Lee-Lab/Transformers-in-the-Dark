import numpy as np
import os
import itertools
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_arms', type=int, required=True)
    parser.add_argument('--max_depth', type=int, required=True)

    args = parser.parse_args()

    num_arms = args.num_arms
    max_depth = args.max_depth

    assert num_arms > 0
    assert max_depth > 0

    operation_tokens = [
        'end_of_sample',
        'start_of_round',
        'end_of_round',
        'start_of_iteration',
        'selected_child_and_then_reward',
        'none',
    ]
    operation_tokens += [
        'tree_traversal',
        'r0d0',
    ]

    arm_tokens = [f'i{index}' for index in range(0, num_arms)]
    new_state_tokens = []

    for current_depth in range(1, max_depth + 1):
        state_tokens = list(itertools.product(arm_tokens, repeat=current_depth))
        state_tokens = [['r0'] + list(state_token) for state_token in state_tokens]

        for state_token in state_tokens:
            new_state_token = [f'{token}d{ind_token}' for ind_token, token in enumerate(state_token)]
            new_state_tokens.append('>'.join(new_state_token))

    num_indices = 200

    reward_tokens = [f'{elem:.2f}' for elem in np.linspace(0, 1, 101)]
    index_tokens = [f'{index}' for index in range(0, num_indices)]

    vocab = sorted(list(set(operation_tokens + new_state_tokens + reward_tokens + index_tokens)))
    vocab_size = len(vocab)

    print('vocab')
    for elem in vocab:
        print(elem)
    print('')
    print(f'vocab size {vocab_size}')
    print('')

    stoi = {
        ch: i for i, ch in enumerate(vocab) 
    }
    itos = {
        i: ch for i, ch in enumerate(vocab)
    }

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return ' '.join([itos[i] for i in l])

    meta = {
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
    }

    path_meta = '../results/metas'
    os.makedirs(path_meta, exist_ok=True)

    with open(os.path.join(path_meta, f'meta_tree_search_{num_arms}_{max_depth}.pkl'), 'wb') as f:
        pickle.dump(meta, f)