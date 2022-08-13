import pandas as pd
from pandas import DataFrame
from utils.common import format_str_datetime

class PortfolioManagement:
    """Portfolio and budget
    >>> for i in range(len(close_prices)):
    >>>     market_prices = close_prices.iloc[i]
    >>>     portfolio.update_status(market_prices)
    >>>     if buy_condition:
    >>>         portfolio.update_status_buy()
    >>>     if sell_condition:
    >>>         portfolio.update_status_sell()
    >>> print(portfolio.portfolio)
    >>> print(portfolio.budget)
    >>> print(portfolio.get_portfolio_history_by_date("2021-05-07"))
    """    
    def __init__(self, config):
        self._portfolio = {}
        self._buy_commission = config["buy_commission"]
        self._sell_commission = config["sell_commission"]
        self._budget = config["initial_equity"] * (1 - config["per_cash"])
        self._t_plus = config["t_plus"]
        self._portfolio_history = {}
        self._budget_history = {}
        self._daily_return = []


    def update_status(self, date, market_prices: DataFrame):
        """Update status for every day. Gọi function này để cập nhật trạng thái sau mỗi ngày (index).

        Args:
            market_prices (DataFrame): Giá đóng cửa của các mã cổ phiếu.
        """        
        daily_return = []
        for symbol in self._portfolio:
            self._portfolio[symbol]["market_price"] = market_prices[symbol]
            self._portfolio[symbol]["avai"] += self._portfolio[symbol]["T{}".format(self._t_plus-1)]
            for i in range(self._t_plus-1, 0, -1):
                self._portfolio[symbol]["T{}".format(i)] = self._portfolio[symbol]["T{}".format(i-1)]
            self._portfolio[symbol]["T0"] = 0
            self._portfolio[symbol]["profit"] = self._cal_profit(symbol)
            self._portfolio[symbol]["max_market_price"] = max(self._portfolio[symbol]["market_price"], self._portfolio[symbol]["max_market_price"])
            daily_return.append(self._cal_daily_return(symbol))
        if len(self._portfolio) == 0:
            self._daily_return.append(0)
        else:
            self._daily_return.append(sum(daily_return)/len(daily_return))

        self._restore(date)


    def update_status_buy(self, date, price, size, symbol):
        price_with_com = price * (1 + self._buy_commission)
        self._budget -= price_with_com * size
        if symbol not in self._portfolio:
            self._init_portfolio(symbol, price, size)
        else:
            prev_total = self._portfolio[symbol]["total"]
            prev_value = self._portfolio[symbol]["value"]
            self._portfolio[symbol]["T0"] += size
            self._portfolio[symbol]["avg_price"] = (prev_value + price * size) / (prev_total + size) 
            self._portfolio[symbol]["total"] += size
            self._portfolio[symbol]["profit"] = self._cal_profit(symbol)

        self._restore(date)        

    
    def update_status_sell(self, date, price, size, symbol):
        price_with_com = price * (1 - self._sell_commission)
        self._budget += price_with_com * size 
        self._portfolio[symbol]["total"] -= size
        self._portfolio[symbol]["avai"] -= size
        self._portfolio[symbol]["profit"] = self._cal_profit(symbol)
        if self._portfolio[symbol]["total"] <= 0:
            del self._portfolio[symbol]
            
        self._restore(date)


    def _restore(self, date):
        date = format_str_datetime(date)
        self._portfolio_history[date] = self._portfolio
        self._budget_history[date] = self._budget


    def get_bought_avg_price(self, symbol):
        return self._portfolio[symbol]["bought_avg_price"]


    def get_max_market_price(self, symbol):
        return self._portfolio[symbol]["max_market_price"]


    def get_bought_stocks(self):
        return list(self._portfolio.keys())


    def get_avai(self, symbol):
        return self._portfolio[symbol]["avai"]
    

    def get_equity_final(self):
        net_value = 0
        for symbol in self._portfolio:
            net_value += (self._portfolio[symbol]["total"] * self._portfolio[symbol]["market_price"]) * (1 - self._sell_commission)
        net_value += self._budget
        return net_value


    def get_portfolio_history_by_date(self, date):
        date = format_str_datetime(date)
        return self._portfolio_history[date]

    def get_final_portfolio(self):
        return DataFrame(self._portfolio.values(), index=self._portfolio.keys()), self._budget


    @property
    def portfolio(self):
        return self._portfolio
    
    @property
    def budget(self):
        return self._budget

    def _cal_profit(self, symbol):
        return (self._portfolio[symbol]["market_price"] - self._portfolio[symbol]["bought_avg_price"]) * self._portfolio[symbol]["total"]

    def _cal_daily_return(self, symbol):
        return (self._portfolio[symbol]["market_price"] - self._portfolio[symbol]["bought_avg_price"]) / self._portfolio[symbol]["bought_avg_price"]

    def _init_portfolio(self, symbol, price, size):
        self._portfolio[symbol] = {}
        self._portfolio[symbol]["avai"] = 0
        self._portfolio[symbol]["T0"] = size
        for i in range(1, self._t_plus):
            self._portfolio[symbol]["T{}".format(i)] = 0
        self._portfolio[symbol]["total"] = size
        self._portfolio[symbol]["market_price"] = price
        self._portfolio[symbol]["bought_avg_price"] = price
        self._portfolio[symbol]["profit"] = 0
        self._portfolio[symbol]["max_market_price"] = price
