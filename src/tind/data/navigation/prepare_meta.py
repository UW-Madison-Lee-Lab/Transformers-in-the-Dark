import numpy as np
import os
import argparse
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)

    args = parser.parse_args()

    width = args.width
    height = args.height

    assert width == height
    assert width > 0

    operation_tokens = [
        'end_of_sample',
        'start_of_round',
        'end_of_round',
        'start_of_iteration',
        'selected_child_and_then_reward',
        'none',
    ]

    width_tokens = [f'x{index}' for index in range(0, width)]
    height_tokens = [f'y{index}' for index in range(0, height)]

    width_height_tokens = []

    for width_token in width_tokens:
        for height_token in height_tokens:
            width_height_tokens.append(f'{width_token}{height_token}')

    if width <= 4:
        num_indices = 200
    else:
        num_indices = 400

    reward_tokens = [f'{elem:.2f}' for elem in np.linspace(0, 1, 101)]
    index_tokens = [f'{index}' for index in range(0, num_indices)]

    vocab = sorted(list(set(operation_tokens + width_height_tokens + reward_tokens + index_tokens)))
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

    with open(os.path.join(path_meta, f'meta_navigation_{width}_{height}.pkl'), 'wb') as f:
        pickle.dump(meta, f)