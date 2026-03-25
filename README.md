# Transformers in the Dark: Navigating Unknown Search Spaces via Bandit Feedback

This is the official repository for "Transformers in the Dark: Navigating Unknown Search Spaces via Bandit Feedback," published in Transactions on Machine Learning Research (TMLR).

* [arXiv preprint]()
* [OpenReview forum](https://openreview.net/forum?id=Jij7zCjVfc)
* [GitHub repository](https://github.com/UW-Madison-Lee-Lab/Transformers-in-the-Dark)

## Installation

This project requires Python 3.11 or later.
You can install the project as follows.

```bash
pip install .
```

For an editable install during development, run the following.

```bash
pip install -e .
```

Both commands install the package and its dependencies.

## Execution

All runnable bash scripts are located in the `scripts` directory.
Move into the `scripts` directory before running them.
Run the scripts in the following order.

For multi-reward tree search, run the following.

```bash
source generate_datasets_tree_search.sh # generate training and validation datasets
source train_tree_search.sh # train Transformer models
source sample_tree_search.sh # sample search trajectories from the trained Transformer models

source run_references_tree_search.sh # run reference search algorithms
```

For multi-reward navigation, run the following.

```bash
source generate_datasets_navigation.sh # generate training and validation datasets
source train_navigation.sh # train Transformer models
source sample_navigation.sh # sample search trajectories from the trained Transformer models

source run_references_navigation.sh # run reference search algorithms
```

## License

It is licensed under the [MIT license](LICENSE).

## Citation

```bibtex
@article{KimJ2026tmlr,
    title = {{Transformers in the Dark}: Navigating Unknown Search Spaces via Bandit Feedback},
    author = {Kim, Jungtaek and Zeng, Thomas and Lin, Ziqian and Lee, Minjae and Lee, Chungpa and Sohn, Jy-yong and Koo, Hyung Il and Lee, Kangwook},
    journal = {Transactions on Machine Learning Research},
    year = {2026}
}