import pandas as pd


class Transactions:
    def __init__(self):
        self._transactions = []

    def append_buy(self, date, symbol, type, bought_price, bought_avg_price, size, equity, note=''):
        self._transactions.append({'date': date,
                                   'symbol': symbol,
                                   'type': type,
                                   'bought_price': bought_price,
                                   'bought_avg_price': bought_avg_price,
                                   'sold_price': None,
                                   'avg_sold_price': None,
                                   'size': size,
                                   'profit_rate': None,
                                   'equity': equity,
                                   'note': ''})


    def append_sell(self, date, symbol, type, bought_price, bought_avg_price, sold_price, sold_avg_price, size, equity, note=''):
        self._transactions.append({'date': date,
                                   'symbol': symbol,
                                   'type': type,
                                   'bought_price': bought_price,
                                   'bought_avg_price': bought_avg_price,
                                   'sold_price': sold_price,
                                   'avg_sold_price': sold_avg_price,
                                   'size': size,
                                   'profit_rate': (sold_avg_price - bought_avg_price) * size / bought_avg_price,
                                   'equity': equity,
                                   'note': ''})

    def get_transactions(self):
        return pd.DataFrame(self._transactions)