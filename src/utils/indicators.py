from stockstats import StockDataFrame as Sdf
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.trend import IchimokuIndicator
import numpy as np

def get_ratio(x, y):
    epsilon = 0.0000001
    return x / abs(y + epsilon) - 1

def ohlcv_ratio(df, features, period):
    df["open_ratio"] = df["open"].pct_change(periods=period)
    df["high_ratio"] = df["high"].pct_change(periods=period)
    df["low_ratio"] = df["low"].pct_change(periods=period)
    df["close_ratio"] = df["close"].pct_change(periods=period)
    df["volume_ratio"] = df["volume"].pct_change(periods=period)
    features.extend(["open_ratio", "high_ratio", "low_ratio", "close_ratio", "volume_ratio"])

    return df, features

def close_ratio(df, features, periods):
    stockstat = Sdf.retype(df.copy())    

    for period in periods:         
        df["close_{}_max_ratio".format(period)] = get_ratio(stockstat["close"], stockstat["close_-{}~0_max".format(period)])
        df["close_{}_min_ratio".format(period)] = get_ratio(stockstat["close"], stockstat["close_-{}~0_min".format(period)])
        features.extend(["close_{}_max_ratio".format(period), "close_{}_min_ratio".format(period)])

    return df, features
    
def volume_ratio(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["volume_{}_max_ratio".format(period)] = get_ratio(stockstat["volume"], stockstat["volume_-{}~0_max".format(period)])
        df["volume_{}_min_ratio".format(period)] = get_ratio(stockstat["volume"], stockstat["volume_-{}~0_min".format(period)])
        features.extend(["volume_{}_max_ratio".format(period), "volume_{}_min_ratio".format(period)])
    
    return df, features

def close_sma(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:
        df["close_sma_{}".format(period)] = stockstat["close_{}_sma".format(period)]
        df["close_sma_{}_ratio".format(period)] = get_ratio(df["close"], stockstat["close_{}_sma".format(period)])
        features.extend(["close_sma_{}_ratio".format(period)])

    df["close_sma_5"] = stockstat["close_5_sma"]
    
    return df, features
    

def volume_sma(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["volume_sma_{}".format(period)] = stockstat["volume_{}_sma".format(period)]
        df["volume_sma_{}_ratio".format(period)] = get_ratio(df["volume"], stockstat["volume_{}_sma".format(period)])
        features.extend(["volume_sma_{}_ratio".format(period)])
    
    return df, features

def close_ema(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["close_ema_{}_ratio".format(period)] = get_ratio(df["close"], stockstat["close_{}_ema".format(period)])
        features.extend(["close_ema_{}_ratio".format(period)])
    
    return df, features
    

def volume_ema(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["volume_ema_{}".format(period)] = stockstat["volume_{}_ema".format(period)]
        df["volume_ema_{}_ratio".format(period)] = get_ratio(df["volume"], stockstat["volume_{}_ema".format(period)])
        features.extend(["volume_ema_{}_ratio".format(period)])
    
    return df, features


def atr(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["atr_{}_ratio".format(period)] = get_ratio(stockstat["atr_{}".format(period)], df["close"])
        features.extend(["atr_{}_ratio".format(period)])
    
    return df, features    

def adx(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["adx_{}_ratio".format(period)] = stockstat["dx_{}_ema".format(period)] / 25 - 1        
        features.extend(["adx_{}_ratio".format(period)])

    return df, features

def kdj(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:         
        df["kdjk_{}".format(period)] = stockstat["kdjk_{}".format(period)] / 50 - 1    
        df["kdjd_{}".format(period)] = stockstat["kdjd_{}".format(period)] / 50 - 1    
        df["kdj_{}_ratio".format(period)] = get_ratio(stockstat["kdjk_{}".format(period)], stockstat["kdjd_{}".format(period)])
        features.extend(["kdjk_{}".format(period), "kdjd_{}".format(period), "kdj_{}_ratio".format(period)])
        
    return df, features

def rsi(df, features, periods):
    stockstat = Sdf.retype(df.copy())
    for period in periods:
        df["rsi_{}".format(period)] = stockstat["rsi_{}".format(period)] / 50 - 1 
        df["rsi_{}_change".format(period)] = (df["rsi_{}".format(period)] - df["rsi_{}".format(period)].shift(3))
        features.extend(["rsi_{}".format(period), "rsi_{}_change".format(period)])
    return df, features


def macd(df, features, short_period=9, medium_period = 12, slow_period = 26):
    
    stockstat = Sdf.retype(df.copy())
    ema_short = 'close_{}_ema'.format(medium_period)
    ema_long = 'close_{}_ema'.format(slow_period)
    ema_signal = 'macd_{}_ema'.format(short_period)
    fast = stockstat[ema_short]
    slow = stockstat[ema_long]
    stockstat['macd'] = fast - slow
    stockstat['macds'] = stockstat[ema_signal]    
    stockstat['macdh'] = (stockstat['macd'] - stockstat['macds'])
    df['macdh_normed'] = stockstat["macdh"] / abs(stockstat['macds'])    
    df['macdh_returned'] = (df['macdh_normed'] - df['macdh_normed'].shift(3))

    features.extend(["macdh_normed", "macdh_returned"])
    return df, features


def mfi(df, features, periods):
    for period in periods:
        mfi_generator = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=period)
        df['mfi_{}'.format(period)] = mfi_generator.money_flow_index() / 50 - 1
        df['mfi_{}_change'.format(period)] = (df['mfi_{}'.format(period)] - df['mfi_{}'.format(period)].shift(3))
        features.extend(["mfi_{}".format(period), 'mfi_{}_change'.format(period)])
    return df, features

def bb(df, features):
    stockstat = Sdf.retype(df.copy())
    df['boll_lb'] = stockstat['boll_lb'] / df['close'] - 1
    df['boll_ub'] = stockstat['boll_ub'] / df['close'] - 1

    features.extend(["boll_lb", "boll_ub"])
    return df, features

def trend_return(df, features, n_step_ahead):
    df['daily_return'] = df['close'].pct_change()
    df['stability'] = df['daily_return'].shift(1).rolling(5).std()
    df['trend_return'] = df['close'].pct_change(periods=n_step_ahead) # n_step_ahead=5 is a week
    df['trend_return'] = df['trend_return'].shift(-n_step_ahead)

    features.extend(["trend_return"])
    return df, features

def trend(df, features, trend_up_threshold, trend_down_threshold):
    df["trend"] = 0
    df.loc[(df['trend_return'] > trend_up_threshold), 'trend'] = 1
    df.loc[(df['trend_return'] < -trend_down_threshold), 'trend'] = 2

    features.extend(["trend"])
    return df, features

def arithmetic_returns(df, features):
    df['open_r'] = df['open'] / df['close'] - 1 # Create arithmetic returns column
    df['high_r'] = df['high'] / df['close'] - 1# Create arithmetic returns column
    df['low_r'] = df['low'] / df['close']  - 1 # Create arithmetic returns column
    df['close_r'] = df['close'].pct_change()  # Create arithmetic returns column
    df['volume_r'] = df['volume'].pct_change()

    features.extend(["open_r", "high_r", "low_r", "close_r", "volume_r"])
    return df, features

def obv(df, features, periods):
    obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    for period in periods:               
        df['obv_{}'.format(period)] = df['obv'].pct_change(periods=period)    
        features.extend(['obv_{}'.format(period)])
    return df, features

def ichimoku(df, features, fast_period, medium_period, slow_period):
    ichimoku = IchimokuIndicator(df['high'], df['low'], fast_period, medium_period, slow_period)
    df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    df['ichimoku_a'] = ichimoku.ichimoku_a()
    df['ichimoku_b'] = ichimoku.ichimoku_b()

    features.extend(['ichimoku_conversion_line', 'ichimoku_base_line', 'ichimoku_a', 'ichimoku_b'])
    return df, features

# Stock Trend Prediction Using Candlestick Charting and Ensemble Machine Learning Techniques with a Novelty Feature Engineering Scheme
def k_line(df, features):
    df['k_line'] = 0
    msk_0 = (df['open'] == df['close']) & (df['open'] == df['low']) & (df['open'] == df['high'])
    msk_1 = (df['open'] == df['close']) & (df['open'] == df['high']) & (df['open'] == df['low'])
    msk_2 = (df['open'] == df['low']) & (df['close'] == df['high'])
    msk_3 = (df['open'] == df['high']) & (df['close'] == df['low'])
    msk_4 = (df['open'] == df['close']) & (df['open'] == df['high']) & (df['low'] < df['close'])
    msk_5 = (df['open'] == df['close']) & (df['open'] == df['low']) & (df['high'] > df['high'])
    msk_6 = (df['open'] == df['close']) & (df['low'] < df['close']) & (df['high'] > df['close'])
    msk_7 = (df['open'] > df['low']) & (df['close'] > df['open']) & (df['close'] == df['high'])
    msk_8 = (df['close'] > df['low']) & (df['open'] > df['close']) & (df['open'] == df['high'])
    msk_9 = (df['open'] == df['low']) & (df['close'] > df['open']) & (df['high'] > df['close'])
    msk_10 = (df['close'] == df['low']) & (df['open'] > df['close']) & (df['high'] > df['open'])
    msk_11 = (df['open'] < df['close']) & (df['low'] < df['open']) & (df['high'] > df['close'])
    msk_12 = (df['open'] > df['close']) & (df['low'] < df['close']) & (df['high'] > df['open'])
    
    df.loc[msk_0, 'k_line'] = 0
    df.loc[msk_1, 'k_line'] = 1
    df.loc[msk_2, 'k_line'] = 2
    df.loc[msk_3, 'k_line'] = 3
    df.loc[msk_4, 'k_line'] = 4
    df.loc[msk_5, 'k_line'] = 5
    df.loc[msk_6, 'k_line'] = 6
    df.loc[msk_7, 'k_line'] = 7
    df.loc[msk_8, 'k_line'] = 8
    df.loc[msk_9, 'k_line'] = 9
    df.loc[msk_10, 'k_line'] = 10
    df.loc[msk_11, 'k_line'] = 11
    df.loc[msk_12, 'k_line'] = 12
    features.extend(['k_line'])
    return df, features

def eight_trigrams(df, features):
    df['high_pre'] = df['high'].shift(1)
    df['low_pre'] = df['low'].shift(1)
    df['close_pre'] = df['close'].shift(1)
    df['open_pre'] = df['open'].shift(1)
    df['eight_trigrams'] = 0
    # bear high
    msk_0 = (df['high'] > df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] > df['low_pre'])
    msk_1 = (df['high'] < df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] < df['low_pre'])
    msk_2 = (df['high'] < df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] > df['low_pre'])
    msk_3 = (df['high'] > df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] > df['low_pre'])
    msk_4 = (df['high'] < df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] < df['low_pre'])
    msk_5 = (df['high'] > df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] < df['low_pre'])
    msk_6 = (df['high'] < df['high_pre']) & (df['close'] < df['close_pre']) & (df['low'] > df['low_pre'])
    msk_7 = (df['high'] > df['high_pre']) & (df['close'] > df['close_pre']) & (df['low'] < df['low_pre'])
    df.loc[msk_0, 'eight_trigrams'] = 0
    df.loc[msk_1, 'eight_trigrams'] = 1
    df.loc[msk_2, 'eight_trigrams'] = 2
    df.loc[msk_3, 'eight_trigrams'] = 3
    df.loc[msk_4, 'eight_trigrams'] = 4
    df.loc[msk_5, 'eight_trigrams'] = 5
    df.loc[msk_6, 'eight_trigrams'] = 6
    df.loc[msk_7, 'eight_trigrams'] = 7
    features.extend(['eight_trigrams'])

    return df, features

def remove_outliers(df, features, threshold = 1000):
    for feat in features:
        df[feat].fillna(threshold, inplace = True)
        df.loc[df[feat] > threshold, feat] = threshold
        df.loc[df[feat] < -threshold, feat] = -threshold
    return df

def rs(source_data, features):
    data_storages = {}
    for ticker in list(source_data.keys()):
        data_storages[ticker] = source_data[ticker]

    milestones = [10, 20, 50, 100]
    weight = [0.1, 0.2, 0.3, 0.4]
    stocks = list(source_data.keys())

    for ticker in stocks:

        changes = [data_storages[ticker]["close"].pct_change(milestones[i]).values * weight[i] for i in range(len(milestones))]
        aggregated_changes = np.array(changes)
        aggregated_changes = np.sum(aggregated_changes, axis = 0)
        data_storages[ticker]["agg_changes"] = aggregated_changes   

    df = data_storages[stocks[0]]
    for index in df.index:
        temp = []
        for ticker in stocks:
            temp.append(data_storages[ticker].loc[index]["agg_changes"])
            rank = np.argsort(temp)
        
        for i in range(len(stocks)):    
            ticker_rs = int(list(rank).index(i) / len(stocks) * 100)
            if index in data_storages[stocks[i]].index:
                data_storages[stocks[i]].loc[index, "rs"] = ticker_rs / 100

    for ticker in stocks:
        data_storages[ticker]["rs_change"] = data_storages[ticker]["rs"] - data_storages[ticker]["rs"].shift(3)
    features.extend(["rs", "rs_change"])
        
    return data_storages, features
