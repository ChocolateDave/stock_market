import gym
import numpy as np

class StockMarketEnv(gym.Env):
    # Changes by minute
    def __init__(self, seed=0):
        self.num_correlated_stocks = 19
        self.num_company_states = 5
        self.start_price = 100.
        self.curr_price = self.start_price
        self.std = 100.
        correlated_stocks = np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks))
        self.curr_state = {'stock_price': np.asarray(self.start_price), 'correlated_stocks': correlated_stocks, 'company_states': np.zeros((self.num_company_states))}
        self.seed = seed
        np.random.seed = seed

    def _get_obs(self):
        return self.curr_state

    def reset(self):
        self.std = 100.
        correlated_stocks = np.random.normal(loc=self.start_price, scale=self.std, size=(self.num_correlated_stocks))
        self.curr_state = {'stock_price': np.asarray(self.start_price), 'correlated_stocks': correlated_stocks, 'company_states': np.zeros((self.num_company_states))}

    def step(self, action: np.array):
        n = action.size[0]
        open_ind = np.random.randint(0, high=n, size=1)
        close_ind = np.random.randint(0, high=n, size=1)
        while close_ind == open_ind:
            close_ind = np.random.randint(0, high=n, size=1)
        open = action[open_ind] # implement high low open close
        close = action[close_ind]
        high = np.max(action)
        low = np.min(action)
        self.curr_state['stock_price'] = np.clip(close, 0, None)
        diff = close - self.curr_price
        diffs = np.random.normal(loc = 0, scale = diff * self.std, size=(self.num_correlated_stocks))
        self.curr_state['correlated_stocks'] += diffs
        self.curr_state['correlated_stocks'] = np.clip(self.curr_state['correlated_stocks'], 0, None)
        

        



