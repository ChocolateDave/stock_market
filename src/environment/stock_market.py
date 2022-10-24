# =============================================================================
# @file   stock_market.py
# @author Maverick Zhang
# @date   Sep-25-22
# =============================================================================
from __future__ import annotations

from typing import Any, Optional, Tuple, Union, Sequence

import gym
from gym.core import ActType, ObsType, Env
from gym.spaces import Box, Tuple as TupleSpace, Discrete
import numpy as np

# TODO (Maverick): market maker agent (maybe not needed)
# TODO (Maverick): HMM


class LogarithmAndSharesActionWrapper(gym.ActionWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env=env)

    def action(self, act: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, Tuple[int]]:
        return (1. + np.log(act[0]), tuple(act[1].flatten()))


class StockMarketEnv(gym.Env):
    """Multi-agent Single-stock Trading Scenario.

    Attributes:
        step_size: market update frequency in minutes, default: 1.
        seed: random generator seed.
    """

    def __init__(self,
                 num_agents: int,
                 num_company: int = 5,
                 num_correlated_stocks: int = 19,
                 num_uncorrelated_stocks: int = 10,
                 max_shares: int = 100000,
                 start_prices: Union[float, Sequence[float]] = 100.0, # 1st value is target stock, next are correlated, final ones are uncorrelated
                 min_budget: float = 100.0,
                 max_budget: float = 10000.0,
                 budget_discount: float = 0.9,
                 step_size: float = 1.0,
                 price_std: float = 100.0, # This may need to be tuned way smaller, possibly below 10
                 noise_std: float = 10.0,
                 worth_of_stocks: float = 0.1,
                 seed: int = 0) -> None:
        super().__init__()

        # Agent Parameters
        self.num_agents = num_agents
        self.num_company = num_company
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.budget_discount = budget_discount
        self.max_shares = max_shares
        self.worth_of_stocks = worth_of_stocks

        # Stock Market Parameters
        self.dt = step_size
        self.start_prices = start_prices
        self.price_std = price_std
        self.noise_std = noise_std

        # Observation and Action spaces
        self.n_correlated_stocks = num_correlated_stocks
        self.n_uncorrelated_stocks = num_uncorrelated_stocks
        self.n_stocks = num_correlated_stocks + num_uncorrelated_stocks + 1
        if not isinstance(start_prices, float):
            assert self.n_stocks == self.start_prices.shape or self.n_stocks == self.start_prices.shape[0]
        self.observation_space = Box(low=1.0,
                                     high=float("inf"),
                                     shape=(1,))
        self.action_space = TupleSpace(
            (Box(low=1.0, # Prices, no log
                 high=float("inf"),
                 shape=(self.num_agents)),
             TupleSpace([Discrete(2 * max_shares + 1, start=-max_shares) for i in range(self.num_agents)])) # Shares to put up
        ) # 2 * max_shares + 1 for negative shares, and for 0 shares.
        self._seed = seed
        self.reset()

        # np.random.seed(seed)
        """self.num_agents = 10
        self.num_correlated_stocks = 19
        self.num_uncorrelated_stocks = 10
        self.num_company_states = 5
        self.start_price = 100.
        self.curr_price = self.start_price
        self.std = 100.
        self.worth_of_stocks = 0.1
        self.timestep = 0
        self.ep_len = 390
        self.noise = 10.
        correlated_stocks = np.clip(np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks)), 1, None)
        uncorrelated_stocks = np.clip(np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_uncorrelated_stocks)), 1, None)
        total_stocks = 1 + self.num_correlated_stocks + self.num_uncorrelated_stocks
        views = np.zeros((self.num_agents, total_stocks), dtype=np.int32)
        views[:, 0] = 1
        views[:, 1:1+self.num_correlated_stocks] = 1
        views[[0, 2, 5, 6, 7, 9], 1 + self.num_correlated_stocks:] = 1
        self.curr_state = {
                            'stock_price': np.asarray(self.start_price), 
                            'correlated_stocks': correlated_stocks, 
                            'uncorrelated_stocks': uncorrelated_stocks,
                            'budgets': np.asarray([100., 100., 100., 100., 100., 1000., 1000., 1000., 1000., 10000.]),
                            'shares_held': np.asarray([500., 500., 500., 500., 500., 500., 500., 500., 500., 500.]),
                            'agent_views': views, # num_agents x num of stocks array, one hot encoded, 1 means that it is viewable, 0 is unviewable, order is studied stock, correlated stocks, uncorrelated stocks
                            'company_states': np.zeros((self.num_company_states)),
                          }"""

    def _get_obs(self):
        out = []
        out.append(self.current_price.reshape(1, -1))
        out.append(self.correlated_stocks.reshape(1, -1))
        out.append(self.uncorrelated_stocks.reshape(1, -1))
        out.append(self.budgets.reshape(1, -1))
        out.append(self.shares.reshape(1, -1))
        out.append(self.valid_mask.reshape(1, -1))
        return np.hstack(out).flatten()

    def reset(self, 
              seed: Optional[int] = None,
              return_info: bool = True):
        if seed:
            self._seed = seed
        self.rng = np.random.default_rng(seed=seed or self._seed)

        # Randomly generate starting stock prices for correlated and uncorrelated stocks
        if isinstance(self.start_prices, float):
            self.current_price = self.start_prices
            other_stocks = np.clip(
                np.random.normal(loc=self.start_prices,
                                scale=self.price_std,
                                size=(self.n_correlated_stocks + self.n_uncorrelated_stocks,)),
                a_min=1., a_max=None # Can never have prices below 1 on stock market
            )
        else:
            self.current_price = self.start_prices[0]
            other_stocks = self.start_prices[1:]
        self.correlated_stocks = other_stocks[:self.n_correlated_stocks]
        self.uncorrelated_stocks = other_stocks[self.n_correlated_stocks:]

        # Randomly create masks for agents 
        self.valid_mask = np.zeros(shape=(self.num_agents, self.n_stocks),
                                   dtype="bool")
        self.valid_mask[:, 1:1+self.n_correlated_stocks] = True
        self.valid_mask[self.rng.choice(2, size=self.num_agents).astype(bool),
                        1 + self.n_correlated_stocks:] = True

        # Starting budgets and shares
        self.budgets = self.min_budget + self.rng.random(
            size=(self.num_agents), dtype="float32") * (
                self.max_budget - self.min_budget)
        self.shares = self.rng.integers(low=1, #TODO: move this to init
                                        high=self.max_shares,
                                        size=(self.num_agents))

        # Randomize utility functions
        self.eta = np.clip(
            np.random.normal(loc=1.5, scale=1.5, size=(self.num_agents,)),
            a_min=0, a_max=10
        )
        def utility(c, eta):
            if eta != 1.:
                return (c ** (1. - eta) - 1.) / (1. - eta)
            else:
                return np.log(c)
        self.CRRA_utility = np.vectorize(utility)

        self.timestep = 0
        self.ep_len = 390
        
        return (np.asarray(self.current_price),
                {
                    "correlated_stocks": self.correlated_stocks,
                    "uncorrelated_stocks": self.uncorrelated_stocks,
                    "budgets": self.budgets,
                    "shares": self.shares,
                    "valid_mask": self.valid_mask,
                })

    def step(self, action: Tuple[np.ndarray, Tuple[int]]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        proposed_prices = action[0]
        proposed_shares = np.asarray(action[1])

        prev_price = self.current_price

        # Perform Market Clearing
        profits, delta_shares, close, volatility = self.clear(
            np.copy(proposed_prices), np.copy(proposed_shares), 
            self.current_price)

        self.current_price = np.clip(close, 1., None)

        # Update Correlated Stocks
        diff = self.current_price - prev_price
        diffs = diff / prev_price * self.correlated_stocks + \
            self.rng.normal(loc=0,
                             scale=self.noise_std * volatility,
                             size=(self.n_correlated_stocks)) # according to Gaussian White Noise, these differences are Gaussians
        self.correlated_stocks += diffs
        self.correlated_stocks = np.clip(
            self.correlated_stocks, 1., None)

        # Update Uncorrelated Stocks
        self.uncorrelated_stocks += self.rng.normal(
            loc=0, scale=self.price_std, size=(self.n_uncorrelated_stocks))
        self.uncorrelated_stocks = np.clip(
            self.uncorrelated_stocks, 1., None)
        self.budgets += profits
        self.shares += delta_shares
        c = 1. + self.budget_discount * \
            self.budgets + self.shares * \
            self.current_price * self.worth_of_stocks

        # Are proposed budgets and shares to sell/buy violating constraints?
        potential_budgets = self.budgets + \
            proposed_prices * (-proposed_shares)
        potential_shares_held = self.shares + \
            proposed_shares
        violations = (potential_budgets < 0.) | (potential_shares_held < 0.)
        rewards = np.where(violations, -100, 0.)
        # Of non violating rewards, calculate their utility
        rewards = np.where(
            rewards >= 0., self.CRRA_utility(c, self.eta), rewards)

        self.timestep += 1
        done = False
        info = {
                    "correlated_stocks": self.correlated_stocks,
                    "uncorrelated_stocks": self.uncorrelated_stocks,
                    "budgets": self.budgets,
                    "shares": self.shares,
                    "valid_mask": self.valid_mask,
                }
        if np.any(rewards < 0.) or self.timestep >= self.ep_len:
            self.current_price, info = self.reset()
            done = True
        return np.asarray(self.current_price), rewards, done, None, info

    # Your broker or clearing institution typically does this in real life
    # action is a price and volume array. Volume must be a nonnegative number
    # Implemented as Immediate or Cancel order (IOC) which is usually default in exchanges
    # Technically, this is screwing over sellers, as there are no market makers here.
    # actions: log prices, share_vol = int, + -> buy, - -> sell, 0 -> hold
    # TODO: Cleanup
    def clear(self, proposed_prices:np.ndarray, proposed_shares:np.ndarray, close:np.ndarray) -> Tuple:
        volatility = 1.
        # the standard deviation of share_prices will determine
        # correlated stock standard deviation
        share_prices = []
        n = self.num_agents

       # Now randomly order each
        b = np.sum(proposed_shares > 0)
        s = np.sum(proposed_shares < 0)
        bid_indices = self.rng.permutation(b)
        seller_indices = self.rng.permutation(s)
        bidder_prices = proposed_prices[proposed_shares > 0]
        seller_prices = proposed_prices[proposed_shares < 0]
        bidder_shares_left = np.copy(proposed_shares[proposed_shares > 0])
        seller_shares_left = np.abs(np.copy(proposed_shares[proposed_shares < 0]))
        bid_profits = np.zeros((b))
        seller_profits = np.zeros((s))
        delta_bid_shares = np.zeros((b))
        delta_ask_shares = np.zeros((s))

        i = 0
        while i < b and seller_indices.size > 0:
            bid_idx = bid_indices[i]
            bid_price, bid_vol = bidder_prices[bid_idx], bidder_shares_left[bid_idx]
            to_delete = []
            m = seller_indices.shape[0]
            for j in range(m):
                ask_idx = seller_indices[j]
                ask_price, ask_vol = seller_prices[ask_idx], seller_shares_left[ask_idx]
                if bid_price >= ask_price:  # this may change when adding market maker
                    close = ask_price  # sellers are technically last in transaction even with market markers
                    share_prices.append(bidder_prices[bid_idx]) # This allows us to calculate market volatility
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
