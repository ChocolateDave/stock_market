import gym
import numpy as np

# TODO: market maker agent (maybe not needed)
# TODO: HMM
class StockMarketEnv(gym.Env):
    # Changes by minute
    def __init__(self, seed=0):
        # add decorrelated stocks?
        self.seed = seed
        np.random.seed(seed)
        self.num_agents = 10
        self.num_correlated_stocks = 19
        self.num_uncorrelated_stocks = 10
        self.num_company_states = 5
        self.start_price = 100.
        self.curr_price = self.start_price
        self.std = 100.
        self.worth_of_stocks = 0.1
        self.timestep = 0.
        self.ep_len = 390
        self.noise = 10.
        correlated_stocks = np.clip(np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks)), 1, None)
        uncorrelated_stocks = np.clip(np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_uncorrelated_stocks)), 1, None)
        self.curr_state = {
                            'stock_price': np.asarray(self.start_price), 
                            'correlated_stocks': correlated_stocks, 
                            'uncorrelated_stocks': uncorrelated_stocks,
                            'company_states': np.zeros((self.num_company_states)),
                            'budgets': np.asarray([100., 100., 100., 100., 100., 1000., 1000., 1000., 1000., 10000.]),
                            'shares_held': np.asarray([500., 500., 500., 500., 500., 500., 500., 500., 500., 500.]),
                          }

    def _get_obs(self):
        return self.curr_state

    def reset(self, seed=None):
        if seed:
            self.seed = seed
        np.random.seed(seed=self.seed)
        self.num_agents = 10
        self.num_correlated_stocks = 19
        self.num_uncorrelated_stocks = 10
        self.num_company_states = 5
        self.start_price = 100.
        self.curr_price = self.start_price
        self.std = 100.
        self.worth_of_stocks = 0.1
        self.timestep = 0.
        self.ep_len = 390
        self.noise = 10.
        correlated_stocks = np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks))
        uncorrelated_stocks = np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_uncorrelated_stocks))
        self.curr_state = {
                            'stock_price': np.asarray(self.start_price), 
                            'correlated_stocks': correlated_stocks, 
                            'uncorrelated_stocks': uncorrelated_stocks,
                            'company_states': np.zeros((self.num_company_states)),
                            'budgets': np.asarray([100., 100., 100., 100., 100., 1000., 1000., 1000., 1000., 10000.]),
                            'shares_held': np.asarray([500., 500., 500., 500., 500., 500., 500., 500., 500., 500.]),
                          }

    def step(self, action: np.array):
        curr_price = self.curr_state['stock_price']
        profits, delta_shares, close, volatility = self.clear(np.copy(action), self.curr_state['stock_price'])
        self.curr_state['stock_price'] = np.clip(close, 0, None)
        diff = close - curr_price
        diffs = diff / curr_price * self.curr_state['correlated_stocks'] + np.random.normal(loc = 0, scale = self.noise * volatility, size=(self.num_correlated_stocks))
        self.curr_state['correlated_stocks'] += diffs
        self.curr_state['correlated_stocks'] = np.clip(self.curr_state['correlated_stocks'], 1, None)
        self.curr_state['uncorrelated_stocks'] += np.random.normal(loc = 0, scale = self.std, size=(self.num_uncorrelated_stocks))
        self.curr_state['uncorrelated_stocks'] = np.clip(self.curr_state['uncorrelated_stocks'], 1, None)
        self.curr_state['budgets'] += profits
        self.curr_state['shares_held'] += delta_shares
        rewards = np.where(np.logical_or(self.curr_state['budgets'] < 0., self.curr_state['shares_held'] < 0.), -10000, 0.)
        c = self.curr_state['budgets'] + self.curr_state['shares_held'] * self.curr_state['stock_price'] * self.worth_of_stocks 
        rewards = np.where(rewards >= 0., np.log(c, where=c > 0.), rewards)
        self.timestep += 1
        if np.any(np.logical_or(self.curr_state['budgets'] < 0., self.curr_state['shares_held'] < 0.)):
            return self.curr_state, rewards, True, None, None
        return self.curr_state, rewards, self.timestep >= self.ep_len, None, None

        # implement hmmlearn

    # Your broker or clearing institution typically does this in real life
    # action is a price and volume array. Volume must be a nonnegative number
    # Implemented as Immediate or Cancel order (IOC) which is usually default in exchanges
    # Technically, this is screwing over sellers, as there are no market makers here.
    # TODO: Cleanup
    def clear(self, action: np.array, close):
        volatility =  1.
        share_prices = [] # the standard deviation of share_prices will determine correlated stock standard deviation
        n, _ = action.shape
        bidders = action[action[:, 0] > 0, :]
        sellers = action[action[:, 0] < 0, :]
        print(bidders, sellers)

        # Now randomly order each 
        b, _ = bidders.shape
        s, _ = sellers.shape
        bid_indices = np.random.permutation(b)
        seller_indices = np.random.permutation(s)
        print(bid_indices, seller_indices)
        bid_profits = np.zeros((b))
        seller_profits = np.zeros((s))
        delta_bid_shares = np.zeros((b))
        delta_ask_shares = np.zeros((s))
        
        i = 0
        while i < b and seller_indices.size > 0:
            bid_idx = bid_indices[i]
            bid_price, bid_vol = bidders[bid_idx, 0], bidders[bid_idx, 1]
            print(bid_price)
            to_delete = []
            m = seller_indices.shape[0]
            for j in range(m):
                ask_idx = seller_indices[j]
                ask_price, ask_vol = sellers[ask_idx, 0], sellers[ask_idx, 1]
                ask_price = np.abs(ask_price)
                if bid_price >= ask_price: # this may change when adding market maker
                    close = np.abs(sellers[ask_idx, 0]) # sellers are technically last in transaction even with market markers
                    share_prices.append(bidders[bid_idx, 0])
                    share_prices.append(np.abs(sellers[ask_idx, 0]))
                    if bid_vol < ask_vol:
                        sellers[ask_idx, 1] -= bid_vol
                        bidders[bid_idx, 1] = 0.
                        delta_bid_shares[bid_idx] += bid_vol
                        delta_ask_shares[ask_idx] -= bid_vol
                        bid_profits[bid_idx] -= bid_vol * bidders[bid_idx, 0]
                        seller_profits[ask_idx] += bid_vol * np.abs(sellers[ask_idx, 0])
                        break
                    elif bid_vol > ask_vol:
                        sellers[ask_idx, 1] = 0.
                        bidders[bid_idx, 1] -= ask_vol
                        delta_bid_shares[bid_idx] += ask_vol
                        delta_ask_shares[ask_idx] -= ask_vol
                        bid_vol -= ask_vol
                        to_delete.append(j)
                        bid_profits[bid_idx] -= ask_vol * bidders[bid_idx, 0]
                        seller_profits[ask_idx] += ask_vol * np.abs(sellers[ask_idx, 0])
                    else:
                        sellers[ask_idx, 1] = 0.
                        bidders[bid_idx, 1] = 0.
                        delta_bid_shares[bid_idx] += bid_vol
                        delta_ask_shares[ask_idx] -= bid_vol
                        to_delete.append(j)
                        bid_profits[bid_idx] -= bid_vol * bidders[bid_idx, 0]
                        seller_profits[ask_idx] += ask_vol * np.abs(sellers[ask_idx, 0])
                        break
            if to_delete:
                seller_indices = np.delete(seller_indices, np.asarray(to_delete))
            i += 1
        
        volatility += np.std(np.asarray(share_prices))

        profits = np.zeros((n))
        delta_shares = np.zeros((n))
        bid_profit_idx = 0
        ask_profit_idx = 0
        for i in range(n):
            if action[i, 0] == 0:
                continue
            elif action[i, 0] > 0:
                profits[i] = bid_profits[bid_profit_idx]
                delta_shares[i] = delta_bid_shares[bid_profit_idx]
                bid_profit_idx += 1
            else:
                profits[i] = seller_profits[ask_profit_idx]
                delta_shares[i] = delta_ask_shares[ask_profit_idx]
                ask_profit_idx += 1
        return profits, delta_shares, close, volatility
        



