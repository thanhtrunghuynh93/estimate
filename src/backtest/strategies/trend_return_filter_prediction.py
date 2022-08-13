from backtest.strategy import Strategy
class TrendReturnFilterPrediction(Strategy):
    def init(self) -> None:
        self._preds = self._preds_df.copy()
        self._gts = self._gts_dict.copy()
        self._buy_trend_return_threshold = self._buy_prob_threshold
        self._sell_trend_return_threshold = self._sell_prob_threshold
        
    def stocks_to_sell(self, index):
        pred_trend_return = self._preds.iloc[index]
        filtered_symbols_by_trend_return = pred_trend_return.index[(pred_trend_return <= self._sell_trend_return_threshold)].tolist()
        possible_stocks_to_sell = filtered_symbols_by_trend_return
        return possible_stocks_to_sell

    def stocks_to_buy(self, index):
        pred_trend_return = self._preds.iloc[index]
        filtered_symbols_by_trend_return = pred_trend_return.index[(pred_trend_return >= self._buy_trend_return_threshold)].tolist()
        possible_stocks_to_buy = filtered_symbols_by_trend_return
        return possible_stocks_to_buy