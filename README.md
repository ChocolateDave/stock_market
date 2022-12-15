# Multi-Agent Reinforcement Learning for Assumption-Free Stock Market Modeling

[![license](https://img.shields.io/badge/license-BSD_3--Clause-gold.svg)](https://github.com/ChocolateDave/a2sos/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/541341881.svg)](https://zenodo.org/badge/latestdoi/541341881)

An assumption-free microscopic stock market model built upon multi-agent reinforcement learning.

This is a final project for the course CS285 Deep Reinforcement Learning, Decision Making, and Control at UC Berkeley. Please see our [final report](docs/final.pdf) for more details about this project.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies locally.

```bash
git clone -b master --depth 1 https://github.com/ChocolateDave/stock_market.git
cd stock_market & pip install -r requirements.txt & pip install -e .
```

## Usage

We provide three scripts for running our codes on different settings.

- If you would like to explore agents trained on different learning rates, please run with

    ```bash
    bash stock_market/scripts/run_stock_market_diff_lr $GPU_ID
    ```

- If you would like to investigate agents trained on different budget discount over time, please run with
  
    ```bash
    bash stock_market/scripts/run_stock_market_budget_discount $GPU_ID
    ```

- If you would like to investigate agents trained on different worth of stock, please run with
  
    ```bash
    bash stock_market/scripts/run_stock_market_worth_of_stocks.sh $GPU_ID
    ```

- Or you can run the base training script for more flexibility

    ```bash
    python stock_market/train.py <$ARGUMENTS>
    ```

## Citation

If you use this source code, please cite it using bibtex as below.

```bibtex
@misc{Maverick2013,
  author = {Maverick, Zhang and Juanwu Lu},
  title = {Multi-Agent Reinforcement Learning for Assumption-Free Stock Market Modeling},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ChocolateDave/stock_market}},
  commit = {2b675bac077bc695048ce0072f254de25c898050}
}
```

## License

This project is licensed under the [BSD 3-Clause License](./LICENSE)