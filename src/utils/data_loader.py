import numpy as np
from utils.common import get_data_baseline
import utils.indicators as indi
import os

class DataLoaderBase():
    def __init__(self, config):
        self._symbols = config["data"]["symbols"]
        self._config = config
        self._his_window = config["data"]["history_window"]
        self._indicators = config["data"]["indicators"]
        self._include_target = config["data"]["include_target"]
        self._target_col = config["data"]["target_col"]
        self._max_slow_period = max([max(indi.values()) for indi in self._indicators.values() if indi.values()])
        self._n_step_ahead = config["data"]["n_step_ahead"]
        self._outlier_threshold = config["data"]["outlier_threshold"]
        self._pretrained_log = config["model"]["pretrained_log"]
        self._rs_dict = {}


    def _preprocess_data(self, source_df_all, ticker, indicators):
        source_df = source_df_all[ticker]
        df = source_df.copy()
        features = []

        if "ohlcv_ratio" in indicators:
            period = indicators["ohlcv_ratio"]["period"]
            df, features = indi.ohlcv_ratio(df, features, period)

        if "close_ratio" in indicators:
            medium_period = indicators["close_ratio"]["medium_period"]
            slow_period = indicators["close_ratio"]["slow_period"]
            df, features = indi.close_ratio(df, features, [medium_period, slow_period])

        if "volume_ratio" in indicators:
            medium_period = indicators["volume_ratio"]["medium_period"]
            slow_period = indicators["volume_ratio"]["slow_period"]
            df, features = indi.volume_ratio(df, features, [medium_period, slow_period])

        if "close_sma" in indicators:
            medium_period = indicators["close_sma"]["medium_period"]
            slow_period = indicators["close_sma"]["slow_period"]
            df, features = indi.close_sma(df, features, [medium_period, slow_period])

        if "volume_sma" in indicators:
            medium_period = indicators["volume_sma"]["medium_period"]
            slow_period = indicators["volume_sma"]["slow_period"]
            df, features = indi.volume_sma(df, features, [medium_period, slow_period])
        
        if "close_ema" in indicators:
            medium_period = indicators["close_ema"]["medium_period"]
            slow_period = indicators["close_ema"]["slow_period"]
            df, features = indi.close_sma(df, features, [medium_period, slow_period])

        if "volume_ema" in indicators:
            medium_period = indicators["volume_ema"]["medium_period"]
            slow_period = indicators["volume_ema"]["slow_period"]
            df, features = indi.volume_sma(df, features, [medium_period, slow_period])

        if "atr" in indicators:
            medium_period = indicators["atr"]["medium_period"]
            slow_period = indicators["atr"]["slow_period"]
            df, features = indi.atr(df, features, [medium_period, slow_period])

        if "adx" in indicators:
            medium_period = indicators["adx"]["medium_period"]
            slow_period = indicators["adx"]["slow_period"]
            df, features = indi.adx(df, features, [medium_period, slow_period])
        
        if "kdj" in indicators:    
            medium_period = indicators["kdj"]["medium_period"]
            slow_period = indicators["kdj"]["slow_period"]
            df, features = indi.kdj(df, features, [medium_period, slow_period])

        if "rsi" in indicators:  
            medium_period = indicators["rsi"]["medium_period"]
            slow_period = indicators["rsi"]["slow_period"]          
            df, features = indi.rsi(df, features, [medium_period, slow_period])

        if "macd" in indicators:    
            medium_period = indicators["macd"]["medium_period"]
            slow_period = indicators["macd"]["slow_period"]        
            df, features = indi.macd(df, features, 9, 12, 26)
        
        if "mfi" in indicators:
            medium_period = indicators["rsi"]["medium_period"]
            slow_period = indicators["rsi"]["slow_period"]
            df, features = indi.mfi(df, features, [medium_period, slow_period])

        if "bb" in indicators:                
            df, features = indi.bb(df, features)

        if "arithmetic_returns" in indicators:
            df, features = indi.arithmetic_returns(df, features)
        
        if "obv" in indicators:
            medium_period = indicators["rsi"]["medium_period"]
            slow_period = indicators["rsi"]["slow_period"]
            df, features = indi.obv(df, features, [medium_period, slow_period])

        if "ichimoku" in indicators:
            fast_period = indicators["ichimoku"]["fast_period"]
            medium_period = indicators["ichimoku"]["medium_period"]
            slow_period = indicators["ichimoku"]["slow_period"]        
            df, features = indi.ichimoku(df, features, fast_period, medium_period, slow_period)
        
        if "k_line" in indicators:
            df, features = indi.k_line(df, features)
        
        if "eight_trigrams" in indicators:
            df, features = indi.eight_trigrams(df, features)
        
        if "trend_return" in indicators:
            df, features = indi.trend_return(df, features, self._n_step_ahead)
        
        if "trend" in indicators: 
            trend_up_threshold = indicators["trend"]["trend_up_threshold"]
            trend_down_threshold = indicators["trend"]["trend_down_threshold"]
            df, features = indi.trend(df, features, trend_up_threshold, trend_down_threshold, self._n_step_ahead)

        if "rs" in indicators:
            if not self._rs_dict:
                self._rs_dict, features = indi.rs(source_df_all, features)
            else:
                features.extend(["rs", "rs_change"])
            df[features[-2]] = self._rs_dict[ticker][features[-2]]
            df[features[-1]] = self._rs_dict[ticker][features[-1]]

        df = indi.remove_outliers(df, features, threshold = self._outlier_threshold)
        # if cut_last:
        df = df[self._max_slow_period:-self._n_step_ahead] # self._max_slow_period: SMA and other indicators, 5: days of a week
        # else:
        #     df = df[self._max_slow_period:]
        if not self._include_target:
            features.remove(self._target_col)
        df = df.interpolate(limit_direction="both")
        df_y = df[self._target_col]
        df_x = df.filter((features))
        return df, df_x, df_y


    def _fill_missing_data_Hieu(self, list_df):
        full_data = max(list_df, key=len)
        max_len = len(full_data) # Get max len of the symbol having full data
        for i in range(len(list_df)):
            if len(list_df[i]) < max_len:
                for count, ind in enumerate(full_data.index):
                    if ind not in list_df[i].index:
                        list_df[i].loc[ind] = list_df[i].loc[full_data.iloc[count-1].name].copy()
                        list_df[i].loc[ind]["open"] = list_df[i].loc[ind]["close"].copy()
            list_df[i].sort_index(inplace=True)
        
        return list_df

    def _fill_missing_data_Trung(self, dict_df):
        tickers = list(dict_df.keys())
        max_daily_len = 0
        max_trade_index = None
        for ticker in tickers:
            source_df = dict_df[ticker]
            if len(source_df) > max_daily_len:
                max_daily_len = len(source_df)
                max_trade_index = source_df.index
                
        for ticker in tickers:    
            missing_points = np.setdiff1d(max_trade_index, dict_df[ticker].index)
            dict_df[ticker].fillna(np.NaN, inplace=True)
            dict_df[ticker] = dict_df[ticker].interpolate(limit_direction = "both")
            # dict_df[ticker].replace(0, np.nan, inplace=True) # Data baseline US có một số volume giá trị bằng 0
            if len(missing_points) > 0:
                # print("{} is missing at: {}".format(ticker, missing_points))
                for ind in missing_points:
                    dict_df[ticker].loc[ind] = np.nan
                dict_df[ticker] = dict_df[ticker].sort_index(axis = 0, ascending = True)
                dict_df[ticker]["open"] = dict_df[ticker]["open"].interpolate(limit_direction = "both")
                dict_df[ticker]["high"] = dict_df[ticker]["high"].interpolate(limit_direction = "both")
                dict_df[ticker]["low"] = dict_df[ticker]["low"].interpolate(limit_direction = "both")
                dict_df[ticker]["close"] = dict_df[ticker]["close"].interpolate(limit_direction = "both")
                dict_df[ticker]["volume"] = dict_df[ticker]["volume"].interpolate(limit_direction = "both")
        return dict_df

    def gen_backtest_data(self, start_date, end_date):
        path_data = "{}/backtest_{}_{}.npz".format(self._pretrained_log, start_date, end_date)
        if not os.path.exists(path_data):
            inputs, gts = self._gen_backtest_data(start_date, end_date)
            np.savez(path_data, inputs=inputs, gts=gts)
        else:
            data = np.load(path_data, allow_pickle=True)
            inputs = data["inputs"]
            gts = data["gts"].item()
        return inputs, gts 

    def _gen_backtest_data(self, start_date, end_date):
        gts = {}
        inputs = []
        inputs_storage = {}
        for sym in self._symbols:
            data = get_data_baseline(sym, start_date, end_date, self._his_window + self._max_slow_period + 5, self._n_step_ahead) # for trend prediction plot
            # data = get_data_baseline(sym, start_date, end_date, self._his_window + self._max_slow_period, self._n_step_ahead)
            inputs_storage[sym] = data
        inputs_storage = self._fill_missing_data_Trung(inputs_storage)
        for sym in inputs_storage:
            df, df_x, _ = self._preprocess_data(inputs_storage, sym, self._indicators)
            df = df.iloc[self._his_window:]
            df_x = df_x.to_numpy()
            gts[sym] = df[["open", "high", "low", "close", "volume", "trend_return"]]
            input_data = np.array([df_x[i: i + self._his_window] for i in range(len(df_x) - self._his_window)])
            inputs.append(input_data)
        num_sample = inputs[0].shape[0]
        inputs = np.vstack((inputs))
        inputs = np.reshape(inputs, (num_sample, len(self._symbols), inputs.shape[1], inputs.shape[2]))
        return inputs, gts