import os
import pickle
import copy
from contextlib import nullcontext
import torch
import numpy as np

from tind import tree_traversal
from tind.llama import MyLlamaConfig, MyLlamaModel, get_allowed_tokens
from tind.utils import get_random_states_test

from tind.algorithms.selection import select_method
from tind.algorithms.utils import get_search_method_best_child_method

from tind.data.tree_search.generate import get_random_tree_search
from tind.data.tree_search.utils import get_file_name, get_rewards


# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
start = "start_of_round" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 100 # number of samples to draw
max_new_tokens = 1 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = True # use PyTorch 2.0 to compile the model to be faster

block_size = 1000000
seed_base = 1337

method_train = 'abcde'
num_instances_train = 1000000
num_goals_train = 1000000
num_goals_test = 1000000
num_arms_train = 1000000
num_arms_test = 1000000
max_depth = 1000000
num_simulations_train = 1000000
num_simulations_test = 1000000
max_tree_depth_train = 1000000
max_tree_depth_test = 1000000

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')).read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

assert block_size != 1000000
assert method_train != 'abcde'
assert num_instances_train != 1000000
assert num_goals_train != 1000000
assert num_goals_test != 1000000
assert num_arms_train != 1000000
assert num_arms_test != 1000000
assert max_depth != 1000000
assert num_simulations_train != 1000000
assert num_simulations_test != 1000000
assert max_tree_depth_train != 1000000
assert max_tree_depth_test != 1000000

str_dataset = 'tree_search'
num_traces = 100
num_rollouts = 10

rewards_train = get_rewards(num_goals_train)
rewards_test = get_rewards(num_goals_test)

concatenate_trajectory = True
fix_num_simulations = True
verbose = False

num_instances_test = 10
random_state_problem, random_state_search = get_random_states_test(seed_base)

out_dir = f"{out_dir}_{str_dataset}_{method_train}_{num_instances_train}_{num_goals_train}_{num_arms_train}_{max_depth}_{num_simulations_train}_{max_tree_depth_train}"
out_dir = f"{out_dir}_{block_size}_{seed_base}"
out_dir = os.path.join("../models", out_dir)


if __name__ == '__main__':
    torch.manual_seed(seed_base)
    torch.cuda.manual_seed(seed_base)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = MyLlamaConfig(**checkpoint['model_args'])
    model = MyLlamaModel(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'

    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    if 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        str_meta = f"meta_{str_dataset}_{num_arms_test}_{max_depth}.pkl"
        meta_path = os.path.join('../results/metas', str_meta)
        assert os.path.exists(meta_path)

    print(f"Loading meta from {meta_path}...")
    print('')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s.split()]
    decode = lambda l: ' '.join([itos[i] for i in l])

    allowed_tokens, all_allowed_tokens = get_allowed_tokens(stoi, meta['vocab_size'])
    assert torch.sum(allowed_tokens) == 200 + 1
    assert torch.sum(all_allowed_tokens) == meta['vocab_size']

    allowed_tokens = allowed_tokens.to(device)
    all_allowed_tokens = all_allowed_tokens.to(device)
    current_allowed_tokens = all_allowed_tokens

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    print('=' * 30)
    print(f'str_dataset {str_dataset}')
    print(f'block_size {block_size}')
    print(f'num_instances_train {num_instances_train}')
    print(f'num_instances_test {num_instances_test}')
    print(f'num_goals_train {num_goals_train}')
    print(f'num_goals_test {num_goals_test}')
    print(f'num_arms_train {num_arms_train}')
    print(f'num_arms_test {num_arms_test}')
    print(f'max_depth {max_depth}')
    print(f'num_traces {num_traces}')
    print(f'num_simulations_train {num_simulations_train}')
    print(f'num_simulations_test {num_simulations_test}')
    print(f'max_tree_depth_train {max_tree_depth_train}')
    print(f'max_tree_depth_test {max_tree_depth_test}')
    print(f'num_rollouts {num_rollouts}')
    print(f'seed_base {seed_base}')
    print(f'fix_num_simulations {fix_num_simulations}')
    print('=' * 30)
    print('')

    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    count = 0
    count_end = 0
    count_goal = 0

    list_num_tokens = []
    list_simulation_goal = []
    list_current_num_simulations = []
    list_minimum_depth_goal = []
    list_steps_to_goals = []
    list_rewards_achieved = []
    list_jump_distances = []
    list_count_behavior_cloning = []

    with torch.no_grad():
        with ctx:
            for ind_instance in range(0, num_instances_test):
                start_state = get_random_tree_search(num_arms_test, max_depth, rewards_test, random_state_problem)
                steps_to_goals = [len(goal) - 1 for goal in start_state.tree_search_spec.goals]

                for k in range(0, num_samples):
                    print('=' * 30)
                    print(f'tree_search {ind_instance} sample {k}')
                    print('=' * 30)
                    print('')

                    count += 1
                    current_num_simulations = 0
                    count_behavior_cloning = 0
                    int_tokens = x.clone()

                    search_node = select_method(
                        method_train, start_state, None, num_simulations_test, max_tree_depth_test, num_rollouts, concatenate_trajectory, random_state_search, verbose)

                    while True:
                        int_tokens = model.generate(int_tokens, max_new_tokens, temperature=temperature, allowed_tokens=current_allowed_tokens, top_k=top_k)

                        try:
                            words = decode(int_tokens[0].tolist())
                        except:
                            break

                        words = words.split()
                        assert int_tokens[0].shape[0] == len(words)

                        all_unexplored_children = tree_traversal.get_all_unexplored_children(search_node)

                        if words[-1] == 'start_of_iteration':
                            current_num_simulations += 1

                            str_all_unexplored_children = tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children, show_full_trajectory=True, concatenate_trajectory=concatenate_trajectory)
                            str_node_info = f'{str_all_unexplored_children} selected_child_and_then_reward'

                            try:
                                int_tokens_node_info = torch.tensor([encode(str_node_info)]).to(device)
                            except:
                                break

                            int_tokens = torch.concat([
                                int_tokens,
                                int_tokens_node_info,
                            ], dim=1)

                            current_allowed_tokens = allowed_tokens
                            continue

                        if len(words) > 1 and words[-2] == 'selected_child_and_then_reward':
                            if words[-1] == 'none':
                                str_reward = f'none'
                                int_tokens_reward = torch.tensor([encode(str_reward)]).to(device)

                                int_tokens = torch.concat([
                                    int_tokens,
                                    int_tokens_reward,
                                ], dim=1)

                                words = decode(int_tokens[0].tolist())
                                words = words.split()
                                assert int_tokens[0].shape[0] == len(words)

                                current_allowed_tokens = all_allowed_tokens
                            elif words[-1].isdigit() and int(words[-1]) < len(all_unexplored_children):
                                index_child = int(words[-1])
                                unexplored_child = all_unexplored_children[index_child]
                                current_depth = len(unexplored_child) - 2 - 1 # exclude 'unexplored' and last node, and root node

                                bhc_node = copy.deepcopy(search_node)
                                trajectory_bhc_node, _ = bhc_node.iteration(0)

                                index_child_bhc_node = tree_traversal.get_trajectory_index(all_unexplored_children, trajectory_bhc_node)
                                if index_child == index_child_bhc_node:
                                    count_behavior_cloning += 1

                                current_node = search_node.traversal(unexplored_child)

                                next_state = unexplored_child[-1]
                                next_node = select_method(
                                    method_train, next_state, current_node, num_simulations_test, max_tree_depth_test, num_rollouts, concatenate_trajectory, random_state_search, verbose)

                                current_node.children.append(next_node)
                                current_node = next_node

                                search_node.metric_updates(current_num_simulations - 1, unexplored_child)
                                reward = search_node.rollouts(current_node, current_depth)

                                current_node.backpropagation(reward)

                                all_unexplored_children = tree_traversal.get_all_unexplored_children(search_node)
                                if 'mcts_' in method_train or 'nmcts_' in method_train or method_train == 'ups':
                                    search_node.fully_explored_update(search_node, all_unexplored_children)

                                str_reward = f'{reward:.2f}'
                                int_tokens_reward = torch.tensor([encode(str_reward)]).to(device)

                                int_tokens = torch.concat([
                                    int_tokens,
                                    int_tokens_reward,
                                ], dim=1)

                                words = decode(int_tokens[0].tolist())
                                words = words.split()
                                assert int_tokens[0].shape[0] == len(words)

                                current_allowed_tokens = all_allowed_tokens
                            else:
                                int_tokens = int_tokens[:, :-1]

                                current_allowed_tokens = allowed_tokens
                                continue

                        if fix_num_simulations:
                            if current_num_simulations >= num_simulations_test:
                                str_end_of_round = 'end_of_round'
                                int_tokens_end_of_round = torch.tensor([encode(str_end_of_round)]).to(device)

                                int_tokens = torch.concat([
                                    int_tokens,
                                    int_tokens_end_of_round,
                                ], dim=1)

                                words = decode(int_tokens[0].tolist())
                                words = words.split()
                                assert int_tokens[0].shape[0] == len(words)

                                current_allowed_tokens = all_allowed_tokens
                            else:
                                str_start_of_iteration = 'start_of_iteration'
                                int_tokens_start_of_iteration = torch.tensor([encode(str_start_of_iteration)]).to(device)

                                str_all_unexplored_children = tree_traversal.convert_all_unexplored_children_to_strings(all_unexplored_children, show_full_trajectory=True, concatenate_trajectory=concatenate_trajectory)
                                str_node_info = f'{str_all_unexplored_children} selected_child_and_then_reward'

                                try:
                                    int_tokens_node_info = torch.tensor([encode(str_node_info)]).to(device)
                                except:
                                    break

                                int_tokens = torch.concat([
                                    int_tokens,
                                    int_tokens_start_of_iteration,
                                    int_tokens_node_info,
                                ], dim=1)

                                current_num_simulations += 1
                                current_allowed_tokens = allowed_tokens
                                continue

                        if len(words) > 1000000:
                            break

                        if words[-1] == 'end_of_round':
                            assert int_tokens.shape[1] == len(words)

                            count_end += 1
                            list_num_tokens.append(len(words))
                            simulation_goal = search_node.simulation_achieving_goal
                            minimum_depth_goal = search_node.minimum_depth_achieving_goal
                            rewards_achieved = search_node.rewards_achieved
                            jump_distances = search_node.jump_distances

                            if simulation_goal != -1:
                                assert minimum_depth_goal != np.inf
                                count_goal += 1
                            if minimum_depth_goal != np.inf:
                                assert simulation_goal != -1

                            list_simulation_goal.append(simulation_goal)
                            list_current_num_simulations.append(current_num_simulations)
                            list_minimum_depth_goal.append(minimum_depth_goal)
                            list_steps_to_goals.append(steps_to_goals)
                            list_rewards_achieved.append(rewards_achieved)
                            list_jump_distances.append(jump_distances)
                            list_count_behavior_cloning.append(count_behavior_cloning)

                            for word in words:
                                print(word)
                            break

                    print('')

    steps_to_best_goal = np.array(list_steps_to_goals)[:, 0]
    minimum_depth_goal = np.array(list_minimum_depth_goal)
    count_behavior_cloning = np.array(list_count_behavior_cloning)
    current_num_simulations = np.array(list_current_num_simulations)

    list_num_tokens = np.array(list_num_tokens)
    list_simulation_goal = np.array(list_simulation_goal)
    list_steps_to_goals = np.array(list_steps_to_goals)
    list_rewards_achieved = np.array(list_rewards_achieved)
    list_jump_distances = np.array(list_jump_distances)

    print(f'count {count}')
    print(f'count_goal {count_goal}')
    print(f'end_rate: {count_end / count:.4f}')
    print(f'goal_rate: {count_goal / count:.4f}')
    print(f'num_tokens: {np.mean(list_num_tokens):.4f} +- {np.std(list_num_tokens):.4f}')
    print(f'simulation_goal: {np.mean(list_simulation_goal):.4f} +- {np.std(list_simulation_goal):.4f}')
    print(f'rewards_achieved: {np.mean(list_rewards_achieved):.4f} +- {np.std(list_rewards_achieved):.4f}')
    print(f'jump_distances: {np.mean(list_jump_distances):.4f} +- {np.std(list_jump_distances):.4f}')
    print(f'num_simulations: {np.mean(current_num_simulations):.4f} +- {np.std(current_num_simulations):.4f}')

    method = 'ours'
    search_method, best_child_method = get_search_method_best_child_method(method_train)

    dict_result = {
        'method': method,
        'search_method': search_method,
        'best_child_method': best_child_method,
        'method_train': method_train,
        'temperature': temperature,
        'num_instances_train': num_instances_train,
        'num_arms_train': num_arms_train,
        'num_arms_test': num_arms_test,
        'max_depth': max_depth,
        'num_goals_train': num_goals_train,
        'num_goals_test': num_goals_test,
        'rewards_train': rewards_train,
        'rewards_test': rewards_test,
        'num_simulations_train': num_simulations_train,
        'num_simulations_test': num_simulations_test,
        'max_tree_depth_train': max_tree_depth_train,
        'max_tree_depth_test': max_tree_depth_test,
        'block_size': block_size,
        'seed_base': seed_base,
        'count': count,
        'count_end': count_end,
        'count_goal': count_goal,
        'end_rate': count_end / count,
        'goal_rate': count_goal / count,
        'num_tokens': list_num_tokens,
        'simulation_goal': list_simulation_goal,
        'current_num_simulations': current_num_simulations,
        'minimum_depth_goal': minimum_depth_goal,
        'steps_to_goals': list_steps_to_goals,
        'rewards_achieved': list_rewards_achieved,
        'jump_distances': list_jump_distances,
        'count_behavior_cloning': count_behavior_cloning,
    }

    path_ours = '../results/ours'
    os.makedirs(path_ours, exist_ok=True)
    str_rewards_train = '_'.join([f'{reward:.1f}' for reward in rewards_train])
    str_rewards_test = '_'.join([f'{reward:.1f}' for reward in rewards_test])
    str_file = f'tree_search_ours_{method_train}_{num_arms_train}_{num_arms_test}_{max_depth}_{num_instances_train}_{str_rewards_train}_{str_rewards_test}_{num_simulations_train}_{num_simulations_test}_{max_tree_depth_train}_{max_tree_depth_test}_{block_size}_{temperature}_{seed_base}.npy'

    np.save(os.path.join(path_ours, str_file), dict_result)