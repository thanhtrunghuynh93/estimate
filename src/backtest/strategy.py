from backtest.order import OrderManagement
from abc import abstractmethod
from backtest.transactions import Transactions
from utils.backtest import lists_difference, lists_intersection
from pandas import DataFrame
import random
from backtest.portfolio import PortfolioManagement
from backtest.order import OrderManagement
from backtest.report import Report
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import glob

class Strategy():
    """A base of a strategy. Defining a strategy by yourself in "src/backtest/strategies/".
    """

    def __init__(self, gts_dict: dict, gts_df: DataFrame, preds_df: DataFrame, config: dict, log_dir: str, buy_prob_threshold: list, sell_prob_threshold: list) -> None:
        """Initialization of Strategy.

        Args:
            strategy (Strategy): A strategy is defined by yourself in "src/backtest/strategies/".
            gts_dict (dict): A dict contains ground-truth prices of each stock, keys are symbols.
            gts_df (DataFrame): A dataframe contains close prices of each stock, column names are symbols.
            preds_df (DataFrame): A dataframe contains preds. Postprocessing in the strategy.
            config (dict): A dict of hyperparameters.
            t_plus (int, optional): Number of days that a stock from the day it was bought 
            come into our accounts (T+ in a stocks market). Defaults to 3 in the Vietnam market.
        """
        self._gts_dict = gts_dict
        self._buy_prob_threshold = buy_prob_threshold
        self._sell_prob_threshold = sell_prob_threshold
        self._gts_df = gts_df
        self._preds_df = preds_df
        self._config = config
        self._log_dir = log_dir
        self._transactions = Transactions()
        self._orders = OrderManagement()
        self._symbols = gts_df.columns
        self._max_stocks = min(self._config["max_stocks"], len(self._symbols))
        self._trailing_sl = self._config["stop_loss"]
        self._buy_commission = self._config["buy_commission"]
        self._sell_commission = self._config["sell_commission"]

    @property
    def transactions(self):
        return self._transactions

    @property
    def portfolio(self):
        return self._portfolio

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def execute(self):
        reports = []
        daily_returns = []
        for runth in tqdm(range(self._config["runs"]), desc="Multi-running backtest"):
            self._portfolio = PortfolioManagement(self._config)
            self._transactions = Transactions()
            self._transactions_plot = {k: pd.DataFrame(np.nan, index=self._gts_df.index, columns=['date', 'buy_signal', 'sell_signal']) for k in self._symbols}

            for i in range(len(self._gts_df)):
                current_date = self._gts_df.index[i]
                current_market_prices = self._gts_df.iloc[i]
                self._portfolio.update_status(
                    current_date, current_market_prices)
                bought_stocks = self._portfolio.get_bought_stocks()
                stocks_to_sell = lists_intersection(self.stocks_to_sell(i), bought_stocks)
                for stock in stocks_to_sell:
                    current_price = self._gts_df[stock].iloc[i]
                    size = self._portfolio.get_avai(stock)
                    if size > 0:
                        self.sell(current_date, current_price, size, stock)

                bought_stocks = self._portfolio.get_bought_stocks()
                if len(bought_stocks) > 0:
                    for stock in bought_stocks:
                        max_market_price = self._portfolio.get_max_market_price(stock)
                        if (max_market_price - current_market_prices[stock]) / max_market_price >= self._trailing_sl:
                            current_price = self._gts_df[stock].iloc[i]
                            size = self._portfolio.get_avai(stock)
                            if size > 0:
                                self.sell(current_date, current_price, size, stock)

                bought_stocks = self._portfolio.get_bought_stocks()
                if len(bought_stocks) < self._max_stocks:
                    num_stocks_to_buy = self._max_stocks - len(bought_stocks)
                    possible_stocks_to_buy = lists_difference(
                        self.stocks_to_buy(i), bought_stocks)
                    possible_stocks_to_buy.sort()
                    if len(possible_stocks_to_buy) > 0:
                        stocks_to_buy = []
                        while len(stocks_to_buy) < min(len(possible_stocks_to_buy), num_stocks_to_buy):
                            ran_stock = random.randint(0, len(possible_stocks_to_buy)-1)
                            stock_to_buy = possible_stocks_to_buy[ran_stock]
                            stocks_to_buy.append(stock_to_buy)
                            possible_stocks_to_buy.remove(stock_to_buy)
                        current_budget = self._portfolio.budget
                        bugdet_each_stock = int(
                            current_budget / len(stocks_to_buy))
                        for stock in stocks_to_buy:
                            current_price = self._gts_df[stock].iloc[i]
                            size = bugdet_each_stock // (current_price * (1 + self._buy_commission)) // 100 * 100
                            if size >= 100:
                                self.buy(current_date, current_price, size, stock)

                if i == len(self._gts_df)-1:
                    for stock in self._portfolio.get_bought_stocks():
                        current_price = self._gts_df[stock].iloc[i]
                        size = self._portfolio.get_avai(stock)
                        if size > 0:
                            self.sell(current_date, current_price, size, stock)

            report = Report(self._portfolio, self._transactions,
                            self._gts_df, self._config)
            result = report.compute_stats()
            reports.append(result)
            daily_returns.append(self._portfolio._daily_return)
            
        daily_return = pd.DataFrame(np.asarray(daily_returns).transpose(), columns=["return_{}".format(i) for i in range(len(daily_returns))])
        daily_return["date"] = self._gts_df.index
        if os.path.exists(os.path.join(self._log_dir, 'daily_returns_1.csv')):
            latest_result_index = max([int(os.path.basename(run_folder).split("_")[2].split(".")[0]) for run_folder in glob.glob(os.path.join(self._log_dir, 'daily_returns*.csv'))])
            result_index = latest_result_index + 1
        else:
            result_index = 1
        daily_return.to_csv(os.path.join(self._log_dir,"daily_returns_{}.csv".format(result_index)), index=False)
        return reports

    @abstractmethod
    def stocks_to_buy(self, ind):
        pass

    @abstractmethod
    def stocks_to_sell(self, ind):
        pass

    def buy(self, date, price, size, symbol):
        self._orders.append_order(date, symbol, "buy", price, size)
        if True:
            self._portfolio.update_status_buy(date, price, size, symbol)
            bought_price = price
            bought_avg_price = self._portfolio.get_bought_avg_price(symbol)
            equity = self._portfolio.get_equity_final()
            self._transactions.append_buy(date, symbol, "buy", bought_price, bought_avg_price, size, equity)
            self._transactions_plot[symbol].loc[date]["buy_signal"] = bought_price

    def sell(self, date, price, size, symbol):
        self._orders.append_order(date, symbol, "sell", price, size)
        if True:
            bought_price = None
            bought_avg_price = self._portfolio.get_bought_avg_price(symbol)
            sold_price = price
            sold_avg_price = price * (1 - self._sell_commission)
            equity = self._portfolio.get_equity_final()
            self._transactions.append_sell(date, symbol, "sell", bought_price, bought_avg_price, sold_price, sold_avg_price, size, equity)
            self._portfolio.update_status_sell(date, price, size, symbol)
            self._transactions_plot[symbol].loc[date]["sell_signal"] = sold_price
