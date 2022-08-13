import pandas as pd
import numpy as np

def sortino_ratio(series, N,rf):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg


def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma


def count_consecutive_pos_values(df):
    m = df.gt(0)
    con_pos_values = ((m != m.shift())[m].cumsum()+1).max()
    if con_pos_values > 0:
        return con_pos_values
    else:
        return 0

def count_consecutive_neg_values(df):
    m = df.lt(0)
    con_pos_values = ((m != m.shift())[m].cumsum()+1).max()
    if con_pos_values > 0:
        return con_pos_values
    else:
        return 0


def lists_intersection(*lists):
    intersection = lists[0]
    for i in range(len(lists)-1):
        intersection = set(intersection).intersection(set(lists[i+1]))
    return list(intersection)

def lists_difference(*lists):
    difference = lists[0]
    for i in range(len(lists)-1):
        difference = set(difference).difference(set(lists[i+1]))
    return list(difference)


def performance_mean(df):
    df_mean = df.mean()
    report = {}
    report["start_date"] = str(df.start_date[0]) 
    report["end_date"] = str(df.end_date[0])
    report["duration"] = int(df_mean.duration)
    report["initial_equity"] = f"{int(df_mean.initial_equity):,}"
    report["cash"] = f"{int(df_mean.cash):,}"
    report["budget"] = f"{int(df_mean.budget):,}"
    report["exposure"] = int(df_mean.exposure)

    report["gross_win"] = f"{int(df_mean.gross_win):,}"
    report["gross_loss"] = f"{int(df_mean.gross_loss):,}"
    report["final_equity"] = f"{int(df_mean.final_equity):,}"
    report["equity_peak"] = f"{int(df_mean.equity_peak):,}"
    report["equity_trough"] = f"{int(df_mean.equity_trough):,}"
    report["profit"] = f"{int(df_mean.profit):,}"
    report["final_return"] = round(df_mean.final_return, 2)
    
    report["acc_return_by_quater"] = _mean_element_of_lists(df.acc_return_by_quater)
    report["return_by_quater"] = _mean_element_of_lists(df.return_by_quater)
    report["buy_and_hold_return"] = int(df_mean.buy_and_hold_return)
    report["max_drawdown"] = f"{int(df_mean.max_drawdown):,}"
    report["max_drawdown_duration"] = int(df_mean.max_drawdown_duration)

    report["sharpe_ratio"] = round(df_mean.sharpe_ratio, 2)
    report["sortino_ratio"] = round(df_mean.sortino_ratio, 2)
    report["calmar_ratio"] = round(df_mean.calmar_ratio, 2)
    report["gain_ratio"] = round(df_mean.gain_ratio, 2)
    report["profit_factor"] = round(df_mean.profit_factor, 2)
    report["expectancy_ratio"] = round(df_mean.expectancy_ratio, 2)
    report["SQN"] = round(df_mean.SQN, 2)

    report["num_buy_trades"] = int(df_mean.num_buy_trades)
    report["num_sell_trades"] = int(df_mean.num_sell_trades)
    report["num_won_trade"] = int(df_mean.num_won_trade)
    report["num_lost_trade"] = int(df_mean.num_lost_trade)
    report["num_consecutive_win_trade"] = int(df_mean.num_consecutive_win_trade)
    report["num_consecutive_lost_trade"] = int(df_mean.num_consecutive_lost_trade)
    report["win_rate"] = round(df_mean.win_rate, 2)
    report["best_trade"] = f"{int(df_mean.best_trade):,}"
    report["worst_trade"] = f"{int(df_mean.worst_trade):,}"
    report["max_trade_duration"] = int(df_mean.max_trade_duration)
    report["avg_trade_duration"] = int(df_mean.avg_trade_duration)

    return report


def _mean_element_of_lists(list_of_lists):
    arrays = [np.array(x) for x in list_of_lists]
    return [np.mean(k) for k in zip(*arrays)]