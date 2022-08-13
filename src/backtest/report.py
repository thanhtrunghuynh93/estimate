from backtest.portfolio import PortfolioManagement
from backtest.transactions import Transactions
from utils.backtest import count_consecutive_pos_values, count_consecutive_neg_values
from utils.common import days_between
import pandas as pd
import numpy as np
import statistics
import math
import warnings
# warnings.filterwarnings("ignore")

class Report:
    """Report. Processings orders.
    """    
    def __init__(self, portfolio: PortfolioManagement, transactions: Transactions, gts_df, config):
        self._report = {}
        self._config = config
        self._transactions = transactions.get_transactions()
        self._portfolio = portfolio
        self._gt_prices = gts_df

        # Basic information
        self._buy_commission = config["buy_commission"]
        self._sell_commission = config["sell_commission"]
        self._start_time = config["start_backtest"]
        self._end_time = config["end_backtest"]
        self._duration = days_between(self._start_time, self._end_time)
        self._initial_equity = config["initial_equity"]
        self._cash = self._initial_equity * config["per_cash"]
        self._budget = self._initial_equity - self._cash
        self._exposure_time = 0

        self._gross_win = 0
        self._gross_loss = 0
        self._equity_final = 0
        self._equity_peak = 0
        self._equity_trough = 0
        self._profit = 0
        self._return_final = 0
        self._acc_return_by_quater = []
        self._return_by_quater = []
        self._votatility = 0 # Chua biet la gi
        self._buy_and_hold_return = 0
        self._max_drawdown = 0
        self._absolute_drawdown = 0
        self._max_drawdown_duration = 0 # max drawdown duration is the worst (the maximum/longest) amount of time an investment has seen between peaks 

        self._sharpe_ratio = 0
        self._sortino_ratio = 0
        self._calmar_ratio = 0
        self._gain_ratio = 0
        self._profit_factor = 0
        self._expectancy_ratio = 0
        self._sqn = 0 # system quality number indicator

        self._num_buy_trades = 0
        self._num_sell_trades = 0
        self._num_won_trade = 0
        self._num_lost_trade = 0
        self._num_consecutive_win_trade = 0
        self._num_consecutive_lost_trade = 0
        self._win_rate = 0
        self._best_trade = 0
        self._worst_trade = 0
        self._max_trade_duration = 0
        self._avg_trade_duration = 0

    
    @property
    def report(self):
        return self._report

    def compute_stats(self):
        # Vì đang lưu thời điểm bán theo cả ngày, nên sẽ không tính được equity cho từng thời điểm bán => ko tính được equity max. Chỉ tính được cho từng ngày.
        if len(self._transactions) > 0:
            all_transactions = self._transactions.copy()
            self._transactions = self._transactions[self._transactions["sold_price"] > 0].reset_index(drop=True)
            self._transactions["profit"] =(self._transactions["sold_price"] - self._transactions["bought_avg_price"]) * self._transactions["size"]
            self._gross_win = self._transactions[self._transactions["profit"] > 0]["profit"].sum()
            self._gross_loss = self._transactions[self._transactions["profit"] < 0]["profit"].sum()
            self._equity_final = self._portfolio.get_equity_final()
            self._equity_peak = self._transactions['equity'].max()
            self._equity_trough = self._transactions['equity'].min()
            self._profit = self._equity_final - self._initial_equity
            self._return_final = self._profit / self._initial_equity
            self._acc_return_by_quater, self._return_by_quater = self._get_return_by_period(self._transactions)
            self._buy_and_hold_return = (self._gt_prices.iloc[-1].mean() - self._gt_prices.iloc[0].mean()) * (self._initial_equity / self._gt_prices.iloc[0].mean())
            self._max_drawdown = self._equity_peak - self._equity_trough
            self._absolute_drawdown = max(0, self._initial_equity - self._equity_trough)
            self._max_drawdown_duration = days_between(self._transactions.iloc[self._transactions['equity'].idxmax()]['date'], self._transactions.iloc[self._transactions['equity'].idxmin()]['date'])
            self._sharpe_ratio = self._get_sharpe_ratio(self._acc_return_by_quater, 0.04)
            self._sortino_ratio = self._get_sortino_ratio(self._acc_return_by_quater, 0.04)
            self._calmar_ratio = self._get_calmar_ratio(self._acc_return_by_quater, self._max_drawdown)
            gain = (self._transactions['sold_price'] - self._transactions['bought_avg_price'])/self._transactions['bought_avg_price']
            self._gain_ratio = gain.mean()
            self._profit_factor = self._gross_win / self._gross_loss
            self._sqn = np.sqrt(len(self._transactions)) * self._transactions['profit'].mean() / self._transactions['profit'].std()

            self._num_buy_trades = len(all_transactions[all_transactions['type'] == 'buy'])
            self._num_sell_trades = len(all_transactions[all_transactions['type'] == 'sell'])
            self._num_won_trade = self._transactions['profit'][self._transactions['profit'] > 0].count()
            self._num_lost_trade = self._transactions['profit'][self._transactions['profit'] < 0].count()
            self._num_consecutive_win_trade = count_consecutive_pos_values(self._transactions['profit'])
            self._num_consecutive_lost_trade = count_consecutive_neg_values(self._transactions['profit'])

            # self._expectancy_ratio = self._num_lost_trade / self._num_won_trade
            self._win_rate = self._num_won_trade / (self._num_lost_trade + self._num_won_trade)
            self._best_trade = self._transactions['profit'].max()
            self._worst_trade = self._transactions['profit'].min()

        self._report = self._format_report()
        return self._report

    def _format_report(self):
        report = {}
        report["start_date"] = str(self._start_time) 
        report["end_date"] = str(self._end_time)
        report["duration"] = int(self._duration)
        report["initial_equity"] = int(self._initial_equity)
        report["cash"] = int(self._cash)
        report["budget"] = int(self._budget)
        report["exposure"] = int(self._exposure_time)

        report["gross_win"] = int(self._gross_win)
        report["gross_loss"] = int(self._gross_loss)
        report["final_equity"] = int(self._equity_final)
        report["equity_peak"] = int(self._equity_peak)
        report["equity_trough"] = int(self._equity_trough)
        report["profit"] = int(self._profit)
        report["final_return"] = round(self._return_final, 2)
        report["acc_return_by_quater"] = self._acc_return_by_quater
        report["return_by_quater"] = self._return_by_quater
        report["buy_and_hold_return"] = int(self._buy_and_hold_return)
        report["max_drawdown"] = int(self._max_drawdown)
        report["absolute_drawdown"] = int(self._absolute_drawdown)
        report["max_drawdown_duration"] = int(self._max_drawdown_duration)

        report["sharpe_ratio"] = round(self._sharpe_ratio, 2)
        report["sortino_ratio"] = round(self._sortino_ratio, 2)
        report["calmar_ratio"] = round(self._calmar_ratio, 2)
        report["gain_ratio"] = round(self._gain_ratio, 2)
        report["profit_factor"] = round(self._profit_factor, 2)
        report["expectancy_ratio"] = round(self._expectancy_ratio, 2)
        report["SQN"] = round(self._sqn, 2)

        report["num_buy_trades"] = int(self._num_buy_trades)
        report["num_sell_trades"] = int(self._num_sell_trades)
        report["num_won_trade"] = int(self._num_won_trade)
        report["num_lost_trade"] = int(self._num_lost_trade)
        report["num_consecutive_win_trade"] = int(self._num_consecutive_win_trade)
        report["num_consecutive_lost_trade"] = int(self._num_consecutive_lost_trade)
        report["win_rate"] = round(self._win_rate, 2)
        report["best_trade"] = int(self._best_trade)
        report["worst_trade"] = int(self._worst_trade)
        report["max_trade_duration"] = int(self._max_trade_duration)
        report["avg_trade_duration"] = int(self._avg_trade_duration)

        return report


    def _get_sharpe_ratio(self, portfolio_return, risk_free_rate):
        excess_return = [rx - risk_free_rate for rx in portfolio_return]
        squared_excess_return = sum([i**2 for i in excess_return])
        return (statistics.mean(portfolio_return) - risk_free_rate) / (math.sqrt(squared_excess_return/len(excess_return)) + 1e-5)


    def _get_sortino_ratio(self, portfolio_return, risk_free_rate):
        excess_return = [rx - risk_free_rate for rx in portfolio_return]
        neg_excess_return = [0 if er > 0 else er for er in excess_return]
        squared_neg_excess_return = sum([i**2 for i in neg_excess_return])
        return (statistics.mean(portfolio_return) - risk_free_rate) / (math.sqrt(squared_neg_excess_return / len(neg_excess_return)) + 1e-5)

    def _get_calmar_ratio(self, portfolio_return, max_drawdown):
        return statistics.mean(portfolio_return) / (max_drawdown / self._equity_peak)
    
    def _get_return_by_period(self, df):
        acc_return_by_quater = []
        return_by_quater = []
        start_index = 0
        df['quarter'] = pd.to_datetime(df['date']).dt.quarter
        df['acc_profit'] = df['profit'].cumsum()
        acc_equity = self._initial_equity
        for i in range(len(df)):
            if (df['quarter'].iloc[start_index] != df['quarter'].iloc[i]) or (i == len(df) - 1):
                acc_profit = df['acc_profit'].iloc[i-1] / acc_equity
                acc_equity += df['acc_profit'].iloc[i-1]
                acc_return_by_quater.append(round(acc_profit, 2))

                profit = df['acc_profit'].iloc[i-1] / self._initial_equity
                return_by_quater.append(round(profit, 2))
                start_index = i
        return acc_return_by_quater, return_by_quater