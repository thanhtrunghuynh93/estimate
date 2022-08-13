import numpy as np
from backtest.report import Report
from backtest.strategy import Strategy
from pandas import DataFrame
import pandas as pd

class Backtest():
    """Backtest
    >>> bt = Backtest(ProbTrendPrediction, gts_dict, preds_df, list_symbols, config)
    >>> report = bt.run_backtest()
    """    
    def __init__(self, strategy: Strategy, gts_dict: dict, preds_df: DataFrame, config: dict, log_dir: str, buy_prob_threshold: list, sell_prob_threshold: list) -> None:
        """Initialization of Backtest.

        Args:
            strategy (Strategy): A strategy is defined by yourself in "src/backtest/strategies/".
            gts_dict (dict): A dict contains ground-truth prices of each stock, keys are symbols.
            preds_df (DataFrame): A dataframe contains preds (columns usually are symbols). 
            Postprocessing in the strategy.
            config (dict): A dict of hyperparameters.
        """        
        self._config = config
        self._preds_df = preds_df
        self._gts_df = self._process_ground_truth(gts_dict)
        self._strategy = strategy(gts_dict, self._gts_df, preds_df, config, log_dir, buy_prob_threshold, sell_prob_threshold)

    def run_backtest(self) -> Report:      
        """Run backtest.

        Returns:
            Report: Return a final report containing multiple metrics/indicators.
        """              
        self._strategy.init()
        report = self._strategy.execute()
        return report

    @property
    def strategy(self):
        return self._strategy

    def _process_ground_truth(self, gts_dict) -> DataFrame:
        gts_df = pd.DataFrame()
        for sym, df in gts_dict.items():
            gt_price = (df["open"] + df["high"]) / 2
            gt_price = pd.DataFrame(gt_price.to_list(), index=gt_price.index, columns=[sym])
            gt_price = gt_price.iloc[:len(self._preds_df)]
            gts_df = pd.concat([gts_df, gt_price], axis=1)
        return gts_df