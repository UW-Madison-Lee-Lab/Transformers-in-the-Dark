import numpy as np
import os
import argparse
import pickle

from tind.data.navigation.utils import get_file_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--num_navigations', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--wall_densities', type=str, required=True)
    parser.add_argument('--num_simulations', type=int, required=True)
    parser.add_argument('--max_tree_depth', type=int, required=True)

    args = parser.parse_args()

    method = args.method
    num_navigations = args.num_navigations
    width = args.width
    height = args.height
    wall_densities = args.wall_densities
    num_simulations = args.num_simulations
    max_tree_depth = args.max_tree_depth

    list_wall_densities = wall_densities.split('-')
    assert len(list_wall_densities) > 1
    assert list_wall_densities[0] == 'wd'

    assert num_navigations > 0
    assert width == height
    assert width > 0
    assert num_simulations > 0
    assert max_tree_depth > 0

    ratio_train = 0.7
    num_traces = 100
    num_rollouts = 10

    str_directory, str_file = get_file_name(
        method, num_navigations, width, height, wall_densities, num_traces, num_simulations, max_tree_depth, num_rollouts)
    list_tokens = []

    with open(os.path.join(str_directory, str_file), 'r') as file_txt:
        for row in file_txt:
            list_tokens.extend(row.split())

    print(list_tokens[:100])
    print(f'len_tokens {len(list_tokens)}')

    path_meta = '../results/metas'

    with open(os.path.join(path_meta, f'meta_navigation_{width}_{height}.pkl'), 'rb') as f:
        meta = pickle.load(f)

    vocab_size = meta['vocab_size']
    stoi = meta['stoi']
    itos = meta['itos']

    def encode(s):
        return [stoi[c] for c in s]

    def decode(l):
        return ' '.join([itos[i] for i in l])

    ind_train = int(len(list_tokens) * ratio_train)
    data_train = list_tokens[:ind_train + 1]
    data_valid = list_tokens[ind_train + 1:]

    ids_train = encode(data_train)
    ids_valid = encode(data_valid)

    print(f'len_ids_train {len(ids_train)}')
    print(f'len_ids_valid {len(ids_valid)}')

    ids_train = np.array(ids_train, dtype=np.uint16)
    ids_valid = np.array(ids_valid, dtype=np.uint16)

    ids_train.tofile(os.path.join(str_directory, 'train.bin'))
    ids_valid.tofile(os.path.join(str_directory, 'val.bin'))