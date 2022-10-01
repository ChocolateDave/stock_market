import gym
import numpy as np

# TODO: market maker agent
# TODO: HMM
# TODO: Use spread to determine next step's volatility
# TODO: implement CRRA utility of budget and shares held, use log
class StockMarketEnv(gym.Env):
    # Changes by minute
    def __init__(self, seed=0):
        # add decorrelated stocks?
        self.seed = seed
        np.random.seed = seed
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

    def _get_obs(self):
        return self.curr_state

    def reset(self):
        self.std = 100.
        correlated_stocks = np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks))
        self.curr_state = {'stock_price': np.asarray(self.start_price), 'correlated_stocks': correlated_stocks, 'company_states': np.zeros((self.num_company_states))}

    def step(self, action: np.array):
        curr_price = self.curr_state['stock_price']
        profits, final_shares, close = self.clear(np.copy(action), self.curr_state['stock_price'])
        self.curr_state['stock_price'] = np.clip(close, 0, None)
        diff = close - curr_price
        diffs = diff / curr_price * self.curr_state['correlated_stocks'] + np.random.normal(loc = 0, scale = self.std, size=(self.num_correlated_stocks))
        self.curr_state['correlated_stocks'] += diffs
        self.curr_state['correlated_stocks'] = np.clip(self.curr_state['correlated_stocks'], 1, None)
        self.curr_state['uncorrelated_stocks'] += np.random.normal(loc = 0, scale = self.std, size=(self.num_uncorrelated_stocks))
        self.curr_state['uncorrelated_stocks'] = np.clip(self.curr_state['uncorrelated_stocks'], 1, None)
        self.curr_state['budgets'] += profits
        self.curr_state['shares_held'] = final_shares
        rewards = np.log(self.curr_state['budgets'] + self.curr_state['shares_held'] * self.curr_state['stock_price'] * self.worth_of_stocks)
        rewards = np.where((profits < 0.) or (final_shares < 0.), -10000, profits)
        self.timestep += 1
        if np.any((profits < 0.) or (final_shares < 0.)):
            return self.curr_state, rewards, True, None, None, None, True
        return self.curr_state, rewards, False, None, None, None, self.timestep >= self.ep_len

        # implement hmmlearn

    # Your broker or clearing institution typically does this in real life
    # action is a price and volume array. Volume must be a nonnegative number
    # Implemented as Immediate or Cancel order (IOC) which is usually default in exchanges
    # Technically, this is screwing over sellers, as there are no market makers here.
    # TODO: Cleanup
    def clear(self, action: np.array, close):
        n, _ = action.shape
        bidders = action[action[:, 0] > 0, :]
        sellers = action[action[:, 0] < 0, :]

        # Now randomly order each 
        b, _ = bidders.shape
        s, _ = sellers.shape
        bid_indices = np.random.permutation(b)
        seller_indices = np.random.permutation(s)
        bid_profits = np.zeros((b))
        seller_profits = np.zeros((s))
        
        i = 0
        while i < b and seller_indices.size > 0:
            bid_idx = bid_indices
            bid_price, bid_vol = bidders[bid_idx]
            to_delete = []
            for j in range(seller_indices.size):
                ask_idx = seller_indices[j]
                ask_price, ask_vol = sellers[ask_idx, :]
                ask_price = np.abs(ask_price)
                if bid_price >= ask_price: # this may change when adding market maker
                    close = np.abs(sellers[ask_idx, 0]) # sellers are technically last in transaction even with market markers
                    if bid_vol < ask_vol:
                        sellers[ask_idx, 1] -= bid_vol
                        bidders[bid_idx, 1] = 0.
                        bid_profits[bid_idx] -= bid_vol * bidders[bid_idx, 0]
                        seller_profits[ask_idx] += bid_vol * np.abs(sellers[ask_idx, 0])
                        break
                    elif bid_vol > ask_vol:
                        sellers[ask_idx, 1] = 0.
                        bidders[bid_idx, 1] -= ask_vol
                        bid_vol -= ask_vol
                        to_delete.append(j)
                        bid_profits[bid_idx] -= ask_vol * bidders[bid_idx, 0]
                        seller_profits[ask_idx] += ask_vol * np.abs(sellers[ask_idx, 0])
                    else:
                        sellers[ask_idx, 1] = 0.
                        bidders[bid_idx, 1] = 0.
                        to_delete.append(j)
                        bid_profits[bid_idx] -= bid_vol * bidders[bid_idx, 0]
                        seller_profits[ask_idx] += ask_vol * np.abs(sellers[ask_idx, 0])
                        break
            np.delete(seller_indices, np.asarray(to_delete))
            i += 1
        
        profits = np.zeros((n))
        final_shares = action[:, 1]
        bid_profit_idx = 0
        ask_profit_idx = 0
        for i in range(n):
            if action[i, 0] == 0:
                continue
            elif action[i, 0] > 0:
                profits[i] = bid_profits[bid_profit_idx]
                final_shares[i] = bidders[bid_profit_idx, 1]
                bid_profit_idx += 1
            else:
                profits[i] = seller_profits[ask_profit_idx]
                final_shares[i] = sellers[ask_profit_idx, 1]
                ask_profit_idx += 1 
        return profits, final_shares, close



        

    # will need two continuous numbers: share price and share volume. Share volume must always be positive integers, so log it and truncate
    # need to add shares into state then. 
    # do I need budget in the state?
    # complete clearing
        



