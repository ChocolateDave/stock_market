# =============================================================================
# @file   stock_market.py
# @author Maverick Zhang
# @date   Sep-25-22
# =============================================================================
import gym
import numpy as np

# TODO (Maverick): market maker agent (maybe not needed)
# TODO (Maverick): HMM


class StockMarketEnv(gym.Env):
    """Multi-agent Single-stock Trading Scenario.

    Attributes:
        step_size: market update frequency in minutes, default: 1.
        seed: random generator seed.
    """

    def __init__(self, step_size: int = 1, seed: int = 0) -> None:
        super().__init__()

        # add decorrelated stocks?
        self.seed = seed
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
        out.append(self.curr_state['stock_price'].reshape(1, -1))
        out.append(self.curr_state['correlated_stocks'].reshape(1, -1))
        out.append(self.curr_state['uncorrelated_stocks'].reshape(1, -1))
        out.append(self.curr_state['budgets'].reshape(1, -1))
        out.append(self.curr_state['shares_held'].reshape(1, -1))
        out.append(self.curr_state['agent_views'].reshape(1, -1))
        out.append(self.curr_state['company_states'].reshape(1, -1))
        return np.hstack(out).flatten()

    def reset(self, seed=None):
        if seed:
            self.seed = seed
        np.random.seed(seed=self.seed)

        def utility(c, eta):
            if eta != 1.:
                return (c ** (1. - eta) - 1.) / (1. - eta)
            else:
                return np.log(c)
        self.CRRA_utility = np.vectorize(utility)
        self.num_agents = 10
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
        self.budget_discount = 0.9
        correlated_stocks = np.clip(np.random.normal(
            loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks)), 1, None)
        uncorrelated_stocks = np.clip(np.random.normal(
            loc=self.start_price, scale=self.std, size=(self.num_uncorrelated_stocks)), 1, None)
        total_stocks = 1 + self.num_correlated_stocks + self.num_uncorrelated_stocks
        self.eta = np.clip(np.random.normal(
            loc=1.5, scale=1.5, size=(self.num_agents)), 0, 10)
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
            'agent_views': views,  # num_agents x num of stocks array, one hot encoded, 1 means that it is viewable, 0 is unviewable, order is studied stock, correlated stocks, uncorrelated stocks
            'company_states': np.zeros((self.num_company_states)),
        }
        return self.curr_state, None

    def step(self, action: np.ndarray):
        proposed_prices = 1. + np.exp(action[:, 0])
        proposed_shares = action[:, 1]
        potential_budgets = self.curr_state['budgets'] + \
            proposed_prices * (-proposed_shares)
        potential_shares_held = self.curr_state['shares_held'] + \
            proposed_shares
        print(potential_budgets, potential_shares_held)
        rewards = np.where(np.logical_or(potential_budgets <
                           0., potential_shares_held < 0.), -100, 0.)
        curr_price = self.curr_state['stock_price']
        profits, delta_shares, close, volatility = self.clear(
            np.copy(action), self.curr_state['stock_price'])
        self.curr_state['stock_price'] = np.clip(close, 0, None)
        diff = close - curr_price
        diffs = diff / curr_price * self.curr_state['correlated_stocks'] + np.random.normal(
            loc=0, scale=self.noise * volatility, size=(self.num_correlated_stocks))
        self.curr_state['correlated_stocks'] += diffs
        self.curr_state['correlated_stocks'] = np.clip(
            self.curr_state['correlated_stocks'], 1, None)
        self.curr_state['uncorrelated_stocks'] += np.random.normal(
            loc=0, scale=self.std, size=(self.num_uncorrelated_stocks))
        self.curr_state['uncorrelated_stocks'] = np.clip(
            self.curr_state['uncorrelated_stocks'], 1, None)
        self.curr_state['budgets'] += profits
        self.curr_state['shares_held'] += delta_shares
        c = 1. + self.budget_discount * \
            self.curr_state['budgets'] + self.curr_state['shares_held'] * \
            self.curr_state['stock_price'] * self.worth_of_stocks
        print(self.eta)
        rewards = np.where(
            rewards >= 0., self.CRRA_utility(c, self.eta), rewards)
        self.timestep += 1
        if np.any(rewards < 0.):
            return self.curr_state, rewards, True, None, None
        return self.curr_state, rewards, self.timestep >= self.ep_len, None, None

        # implement hmmlearn

    # Your broker or clearing institution typically does this in real life
    # action is a price and volume array. Volume must be a nonnegative number
    # Implemented as Immediate or Cancel order (IOC) which is usually default in exchanges
    # Technically, this is screwing over sellers, as there are no market makers here.
    # actions: log prices, share_vol = int, + -> buy, - -> sell, 0 -> hold
    # TODO: Cleanup
    def clear(self, action: np.ndarray, close):
        volatility = 1.
        # the standard deviation of share_prices will determine correlated stock standard deviation
        share_prices = []
        n, _ = action.shape
        bidders = action[action[:, 1] > 0, :]
        sellers = action[action[:, 1] < 0, :]

       # Now randomly order each
        b, _ = bidders.shape
        s, _ = sellers.shape
        bid_indices = np.random.permutation(b)
        seller_indices = np.random.permutation(s)
        bidder_prices = 1. + np.exp(bidders[:, 0])
        seller_prices = 1. + np.exp(sellers[:, 0])
        bidder_shares_left = np.copy(bidders[:, 1])
        seller_shares_left = np.abs(np.copy(sellers[:, 1]))
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
                    share_prices.append(bidders[bid_idx, 0])
                    share_prices.append(sellers[ask_idx, 0])
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
            if action[i, 1] == 0:
                continue
            elif action[i, 1] > 0:
                profits[i] = bid_profits[bid_profit_idx]
                delta_shares[i] = delta_bid_shares[bid_profit_idx]
                bid_profit_idx += 1
            else:
                profits[i] = seller_profits[ask_profit_idx]
                delta_shares[i] = delta_ask_shares[ask_profit_idx]
                ask_profit_idx += 1
        return profits, delta_shares, close, volatility
