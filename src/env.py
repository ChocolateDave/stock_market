# =============================================================================
# @file   stock_market.py
# @author Maverick Zhang
# @date   Sep-25-22
# =============================================================================
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from gymnasium.spaces import Tuple as TupleSpace
from gymnasium.utils import seeding
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWraper
from src.types import OptInt

# TODO (Maverick): market maker agent (maybe not needed)
# TODO (Maverick): HMM


class LogarithmAndIntActionWrapper(BaseParallelWraper):

    def __init__(self, env: ParallelEnv) -> None:
        super().__init__(env=env)

    def step(self,
             actions: Dict[str, Tuple[np.ndarray, int]]
             ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        actions = {
            agent: (
                np.exp(np.arctanh(ac[0])) + 1.0,              # price
                math.ceil(self.env.max_shares * ac[1] - 0.5)  # share volume
            )
            for agent, ac in actions.items()
        }
        return super().step(actions)


class StockMarketEnv(ParallelEnv):
    """Multi-agent Single-stock Trading Scenario.

    Attributes:
        step_size: market update frequency in minutes, default: 1.
        seed: random generator seed.
    """
    metadata: dict[str, any] = {'render_modes': ['human'],
                                'name': 'stock_market_v1'}

    def __init__(self,
                 num_agents: int,
                 max_cycles: int = 390,
                 num_company: int = 5,
                 num_correlated_stocks: int = 19,
                 num_uncorrelated_stocks: int = 10,
                 max_shares: int = 100000,
                 start_prices: Union[float, Sequence[float]] = 100.0,
                 budget_range: Tuple[float, float] = (100.0, 10000.0),
                 budget_discount: float = 0.9,
                 step_size: float = 1.0,
                 price_std: float = 100.0,
                 noise_std: float = 10.0,
                 worth_of_stocks: float = 0.1,
                 seed: int = 42,
                 **kwargs) -> None:
        super().__init__()

        # Agent Parameters
        self.num_company = num_company
        self.budge_range = budget_range
        self.budget_discount = budget_discount
        self.max_shares = max_shares
        self.worth_of_stocks = worth_of_stocks
        self.possible_agents = [f'agent_{i}' for i in range(num_agents)]

        # Stock Market Parameters
        self.dt = step_size
        self.max_cycles = max_cycles
        self.start_prices = start_prices
        self.price_std = price_std
        self.noise_std = noise_std
        self.n_correlated_stocks = num_correlated_stocks
        self.n_uncorrelated_stocks = num_uncorrelated_stocks
        self.n_stocks = num_correlated_stocks + num_uncorrelated_stocks + 1
        if not isinstance(start_prices, float):
            assert self.n_stocks == len(self.start_prices)
        self.seed(seed=seed)
        self._reset_market()

        # Observation and Action spaces
        self._observation_spaces: Dict[str, Any] = {}
        self._action_spaces: Dict[str, Any] = {}
        for agent in self.agents:
            self._observation_spaces[agent] = Box(
                low=1.0,
                high=+float('inf'),
                shape=(self.n_stocks,),
                dtype=np.float32
            )
            self._action_spaces[agent] = TupleSpace((
                Box(low=1.0, high=+float('inf'), shape=(1, )),
                Discrete(2 * max_shares + 1, start=-max_shares)
            ))
        self._state_space = Box(
            low=1.0,
            high=+float('inf'),
            shape=(self.n_stocks, ),
            dtype=np.float32
        )

    def observation_space(self, agent: str) -> Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        return self._action_spaces[agent]

    def reset(self,
              seed: OptInt = None,
              return_info: bool = True,
              options: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
        if seed is not None:
            self.seed(seed=seed)
        self._reset_market()

        return {
            agent: self.state() * self.valid_mask[agent]
            for agent in self.agents
        }

    def seed(self, seed: Optional[seed] = None) -> None:
        self._np_rng, seed = seeding.np_random(seed)

    def state(self) -> np.ndarray:
        return np.hstack([
            self.current_price,
            self.correlated_stocks,
            self.uncorrelated_stocks
        ])

    @property
    def state_space(self) -> Box:
        return self._state_space

    def step(self,
             actions: Dict[str, Tuple[np.ndarray, int]]
             ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        proposed_prices = np.hstack([ac[0] for ac in actions.values()])
        proposed_shares = np.asarray([ac[1] for ac in actions.values()],
                                     dtype="float32")
        prev_price = self.current_price

        # Perform Market Clearing
        profits, delta_shares, close, volatility = self._clear(
            np.copy(proposed_prices), np.copy(proposed_shares),
            self.current_price
        )
        self.current_price = np.clip(close, 1., None)

        # Update Correlated Stocks
        diff = self.current_price - prev_price
        # NOTE: Gaussian White Noise
        diffs = diff / prev_price * self.correlated_stocks + \
            self._np_rng.normal(loc=0,
                                scale=self.noise_std * volatility,
                                size=(self.n_correlated_stocks))
        self.correlated_stocks = np.clip(
            self.correlated_stocks + diffs,
            a_min=1.0, a_max=None
        )

        # Update Uncorrelated Stocks
        self.uncorrelated_stocks = np.clip(
            self.uncorrelated_stocks + self._np_rng.normal(
                loc=0.0, scale=self.price_std,
                size=(self.n_uncorrelated_stocks)
            ),
            a_min=1.0, a_max=None
        )
        self.budgets = self.budgets + profits
        self.shares = self.shares + delta_shares
        c = 1. + self.budget_discount * \
            self.budgets + self.shares * \
            self.current_price * self.worth_of_stocks

        # Are proposed budgets and shares to sell/buy violating constraints?
        potential_budgets = self.budgets - proposed_prices * proposed_shares
        potential_shares_held = self.shares + proposed_shares
        violations = (potential_budgets < 0.) + (potential_shares_held < 0.)
        rewards_n = {
            agent: -100.0 if violate else self.utility(c[i], self.eta[i])
            for i, (agent, violate) in enumerate(zip(self.agents, violations))
        }

        # Retreive values
        self.timestep += 1
        next_obs_n = {agent: self.state() * self.valid_mask[agent]
                      for agent in self.agents}
        dones_n = {agent: True if rewards_n[agent] < 0.0 else False
                   for agent in self.agents}
        env_truncation = self.timestep >= self.max_cycles or \
            all(dones_n.values())  # NOTE: terminates when all are done
        if env_truncation:
            self.agents = []
        truncated_n = {agent: env_truncation for agent in self.agents}
        info = {agent: {} for agent in self.agents}

        return next_obs_n, rewards_n, dones_n, truncated_n, info

    # NOTE: Your broker or clearing institution typically does this in real
    # life action is a price and volume array. Volume must be a nonnegative
    # number implemented as Immediate or Cancel order (IOC)which is usually
    # default in exchanges. Technically, this is screwing over sellers,
    # as there are no market makers here.
    # actions: prices, share_vol = int, + -> buy, - -> sell, 0 -> hold
    # TODO: Cleanup
    def _clear(self,
               proposed_prices: np.ndarray,
               proposed_shares: np.ndarray,
               close: np.ndarray) -> Tuple:
        # The standard deviation of share_prices will determine
        # correlated stock standard deviation
        volatility = 1.
        share_prices = []
        n = self.num_agents

        # Now randomly order each
        b = np.sum(proposed_shares > 0)
        s = np.sum(proposed_shares < 0)
        bid_indices = self._np_rng.permutation(b)
        seller_indices = self._np_rng.permutation(s)
        bidder_prices = proposed_prices[proposed_shares > 0]
        seller_prices = proposed_prices[proposed_shares < 0]
        bidder_shares_left = np.copy(proposed_shares[proposed_shares > 0])
        seller_shares_left = np.abs(
            np.copy(proposed_shares[proposed_shares < 0]))
        bid_profits = np.zeros((b))
        seller_profits = np.zeros((s))
        delta_bid_shares = np.zeros((b))
        delta_ask_shares = np.zeros((s))

        i = 0
        while i < b and seller_indices.size > 0:
            bid_idx = bid_indices[i]
            bid_price, bid_vol = bidder_prices[bid_idx], \
                bidder_shares_left[bid_idx]
            to_delete = []
            m = seller_indices.shape[0]
            for j in range(m):
                ask_idx = seller_indices[j]
                ask_price, ask_vol = seller_prices[ask_idx], \
                    seller_shares_left[ask_idx]
                # NOTE: this may change when adding market maker
                if bid_price >= ask_price:
                    # NOTE: Sellers are technically last in transaction
                    # even with market markers
                    close = ask_price
                    # This allows us to calculate market volatility
                    share_prices.append(bidder_prices[bid_idx])
                    share_prices.append(seller_prices[ask_idx])
                    if bid_vol < ask_vol:
                        seller_shares_left[ask_idx] -= bid_vol
                        bidder_shares_left[bid_idx] = 0.
                        delta_bid_shares[bid_idx] += bid_vol
                        delta_ask_shares[ask_idx] -= bid_vol
                        bid_profits[bid_idx] -= bid_vol * bid_price
                        seller_profits[ask_idx] += bid_vol * ask_price
                        break
                    elif bid_vol > ask_vol:
                        seller_shares_left[ask_idx] = 0.
                        bidder_shares_left[bid_idx] -= ask_vol
                        delta_bid_shares[bid_idx] += ask_vol
                        delta_ask_shares[ask_idx] -= ask_vol
                        bid_vol -= ask_vol
                        to_delete.append(j)
                        bid_profits[bid_idx] -= ask_vol * bid_price
                        seller_profits[ask_idx] += ask_vol * ask_price
                    else:
                        seller_shares_left[ask_idx] = 0.
                        bidder_shares_left[bid_idx] = 0.
                        delta_bid_shares[bid_idx] += bid_vol
                        delta_ask_shares[ask_idx] -= ask_vol
                        to_delete.append(j)
                        bid_profits[bid_idx] -= bid_vol * bid_price
                        seller_profits[ask_idx] += ask_vol * ask_price
                        break
            if to_delete:
                seller_indices = np.delete(
                    seller_indices, np.asarray(to_delete))
            i += 1

        if share_prices:
            volatility += np.std(np.asarray(share_prices))

        profits = np.zeros((n))
        delta_shares = np.zeros((n))
        bid_profit_idx = 0
        ask_profit_idx = 0
        for i in range(n):
            if proposed_shares[i] == 0:
                continue
            elif proposed_shares[i] > 0:
                profits[i] = bid_profits[bid_profit_idx]
                delta_shares[i] = delta_bid_shares[bid_profit_idx]
                bid_profit_idx += 1
            else:
                profits[i] = seller_profits[ask_profit_idx]
                delta_shares[i] = delta_ask_shares[ask_profit_idx]
                ask_profit_idx += 1
        return profits, delta_shares, np.asarray(close), volatility

    def _reset_market(self) -> None:
        # Initialize agents
        self.agents = self.possible_agents[:]
        self._index_map = {
            name: idx for idx, name in enumerate(self.possible_agents)
        }
        self._agent_selector = agent_selector(self.agents)

        # Randomly generate starting stock prices for correlated and
        # uncorrelated stocks. NOTE: minimum stock price is $1.0
        if isinstance(self.start_prices, float):
            self.current_price = self.start_prices
            other_stocks = np.clip(
                np.random.normal(loc=self.start_prices,
                                 scale=self.price_std,
                                 size=(self.n_correlated_stocks +
                                       self.n_uncorrelated_stocks,)),
                a_min=1.0, a_max=None
            )
        else:
            self.current_price = self.start_prices[0]
            other_stocks = self.start_prices[1:]
        self.correlated_stocks = other_stocks[:self.n_correlated_stocks]
        self.uncorrelated_stocks = other_stocks[self.n_correlated_stocks:]

        # Randomly create masks for agents
        self.valid_mask = {agent: np.zeros([self.n_stocks, ], dtype='bool')
                           for agent in self.agents}
        _idcs = self._np_rng.choice(2, self.num_agents).astype('bool')
        for _idx in _idcs.nonzero()[0]:
            self.valid_mask[self.agents[_idx]][
                1:1+self.n_correlated_stocks] = True
            self.valid_mask[self.agents[_idx]][
                -self.n_uncorrelated_stocks:] = True

        # Starting budgets and shares
        self.budgets = self.budge_range[0] + self._np_rng.random(
            size=(self.num_agents), dtype='float32'
        ) * (self.budge_range[1] - self.budge_range[0])
        self.shares = self._np_rng.integers(low=1,  # TODO: move this to init
                                            high=self.max_shares,
                                            size=(self.num_agents))

        # Randomize utility functions
        self.eta = np.clip(
            np.random.normal(loc=1.5, scale=1.5, size=(self.num_agents,)),
            a_min=0, a_max=10
        )
        # self.eta = np.ones(shape=(self.num_agents, )) * 0.5

        self.timestep = 0

    @staticmethod
    def utility(c: float, eta: float, eps: float = 1e-6) -> float:
        if eta != 1.:
            return (c ** (1. - eta) - 1.) / (1. - eta + eps)
        else:
            return math.log(c)