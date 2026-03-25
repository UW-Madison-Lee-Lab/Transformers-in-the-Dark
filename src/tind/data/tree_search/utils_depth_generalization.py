import re


def generate_tree_paths(num_arms: int, max_depth: int) -> list:
    def dfs(path, depth):
        if depth > max_depth:
            return
        if depth > 0:
            paths.append(">".join(path))
        if depth == max_depth:
            return
        for i in range(num_arms):
            dfs(path + [f"i{i}"], depth + 1)

    paths = []
    dfs(["r0"], 0)
    return paths


def reindex_depth_reduced_tree(
        meta: dict,
        depth_org :int,
        depth_reduced:int,
        num_arms :int = 2
    ) -> list:
    assert depth_org > depth_reduced

    stoi = meta['stoi']
    stoi_without_depth = {re.sub(r'd\d+', '', k): v for k, v in stoi.items() if k.startswith("r0d0>")}

    path_from = generate_tree_paths(num_arms, depth_reduced)
    index_from = [stoi_without_depth[path] for path in path_from]

    path_from = [path.lstrip('r0') for path in path_from]
    path_to_roots = generate_tree_paths(num_arms, depth_org - depth_reduced)
    index_to_list = [[stoi_without_depth[path_root+path] for path in path_from] for path_root in path_to_roots]

    return [dict(zip(index_from, index_to)) for index_to in index_to_list]
