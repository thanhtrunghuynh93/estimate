import pandas as pd

class OrderManagement:
    def __init__(self):
        self._orders = []

    def append_order(self,
                date:  str,
                symbol: str,
                type: str,
                price: float,
                size: int,
                ):
        """Append an order.

        Args:
            date (str): Buy/sell date.
            price (float): Buy/sell price.
            size (int): Number of stocks bought/sold.
            symbol (str): Symbol bought/sold.
        """    
        self._orders.append({'date': date,  'symbol': symbol, 'type': type, 'price': price, 'size': size})