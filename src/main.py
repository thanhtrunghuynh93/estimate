import os
import sys
import argparse
import yaml
import glob
import numpy as np
import pandas as pd
from utils.common import days_between, add_days_to_string_time, update_config_to_yaml, seed, get_class
from datetime import datetime


def run_model(config, seed_num):
    model = get_class(os.path.join(args.model_path_folder, config["model"]["path"]))(config)
    test_performances = []
    if args.train:
        model.train()
    if args.test:
        for i in range(args.num_testing):
            seed(seed_num+i)
            test_performances.append(model.test())
    return test_performances

def get_symbols_sp500():
    symbols = np.load("data/US/sp500/baseline_data_sp500.npy", allow_pickle=True).item()
    return list(map(str, symbols.keys()))

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default="estimate")
    parser.add_argument('--model_path_folder',
                        type=str,
                        default="model/estimate/framework/")
    parser.add_argument('--train',
                        type=int,
                        default=1)
    parser.add_argument('--test',
                        type=int,
                        default=1)
    parser.add_argument('--config',
                        type=str,
                        default="model/estimate/config/final.yaml")
    parser.add_argument('--seed',
                        type=int,
                        default=52)
    parser.add_argument('--backtest_period',
                        type=int,
                        default=163)
    parser.add_argument('--start_date_backtest',
                        type=str,
                        default="2017-01-01")
    parser.add_argument('--end_date_backtest',
                        type=str,
                        default="2022-05-01")
    parser.add_argument('--start_train',
                        type=str,
                        default="2016-01-01")
    parser.add_argument('--start_test',
                        type=str,
                        default="2016-11-01")
    parser.add_argument('--run_name', 
                        type=str
                        )

    # Read config
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["model_name"] = args.model
    config["args"] = vars(args)
    config["data"]["symbols"] = get_symbols_sp500()
    seed_num = args.seed
    base_pretrained_folder = "pretrained/{}".format(args.model)
    base_pretrained_run_folder = "pretrained/{}/{}".format(args.model, config["project"]["run_name"])
    if os.path.exists(base_pretrained_folder):
        latest_run_index = max([int(os.path.basename(run_folder).split("-")[1]) for run_folder in glob.glob(os.path.join(base_pretrained_folder, '*'))])
    else:
        latest_run_index = 0
    seed(seed_num)
    run_index = latest_run_index+1
    pretrained_folder = base_pretrained_run_folder + "-{}".format(run_index)
    if args.run_name:
        pretrained_folder = "pretrained/{}/{}".format(args.model, args.run_name)
    backtest_config_baseline = "backtest/config/rolling.yaml"
    with open(backtest_config_baseline) as f:
        backtest_config = yaml.safe_load(f)
    config["backtest"]["config_path"] = backtest_config_baseline
    args.num_testing = 1
    backtest_period = args.backtest_period
    input_start_backtest = args.start_date_backtest
    input_end_backtest = args.end_date_backtest
    start_train = args.start_train
    start_test = args.start_test
    end_train = add_days_to_string_time(start_test, -1)
    end_test = add_days_to_string_time(input_start_backtest, -1)
    flag_break = False
    start_backtest = input_start_backtest
    if days_between(input_start_backtest, input_end_backtest) < backtest_period:
        end_backtest = input_end_backtest
    else:
        end_backtest = add_days_to_string_time(
            input_start_backtest, backtest_period)
    count = 0
    while True:
        count += 1
        rb_pretrained_folder = os.path.join(pretrained_folder, "rb_{}".format(count))
        if not os.path.exists(rb_pretrained_folder):
            os.makedirs(rb_pretrained_folder)
        config["model"]["pretrained_log"] = rb_pretrained_folder
        config["data"]["start_train"] = start_train
        config["data"]["end_train"] = end_train
        config["data"]["start_test"] = start_test
        config["data"]["end_test"] = end_test
        backtest_config["start_backtest"] = start_backtest
        backtest_config["end_backtest"] = end_backtest
        update_config_to_yaml(backtest_config, backtest_config_baseline)
        performances = run_model(config, seed_num)
        for test_th, test_performance in enumerate(performances):
            res = pd.DataFrame([dict(
                date = datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                phase="Phase{}".format(count),
                start_train=start_train,
                end_train=end_train,
                start_test=start_test,
                end_test=end_test,
                start_backtest=start_backtest,
                end_backtest=end_backtest,
                final_return = test_performance[0]["final_return"].mean(),
                final_return_std = test_performance[0]["final_return"].std(),
                sharpe_ratio = test_performance[0]["sharpe_ratio"].mean(),
                mae = test_performance[1]["mae"],
                rmse = test_performance[1]["rmse"],
                long_acc = test_performance[1]["long_acc"],
                short_acc = test_performance[1]["short_acc"],
                ic = test_performance[1]["ic"],
                rank_ic = test_performance[1]["rank_ic"],
                icir = test_performance[1]["icir"],
                rank_icir = test_performance[1]["rank_icir"],
                long_k_prec = test_performance[1]["long_k_prec"],
                short_k_prec = test_performance[1]["short_k_prec"],
            )])
            if os.path.exists(os.path.join(rb_pretrained_folder, 'results_1.csv')):
                latest_result_index = max([int(os.path.basename(run_folder).split("_")[1].split(".")[0]) for run_folder in glob.glob(os.path.join(rb_pretrained_folder, 'results*.csv'))])
                result_index = latest_result_index + 1
            else:
                result_index = 1
            res.to_csv(os.path.join(rb_pretrained_folder,"results_{}.csv".format(result_index)), index=False)
        if flag_break:
            break
        if datetime.strptime(add_days_to_string_time(end_backtest, backtest_period), "%Y-%m-%d") <= datetime.strptime(input_end_backtest, "%Y-%m-%d"):
            start_train = add_days_to_string_time(start_train, backtest_period)
            end_train = add_days_to_string_time(end_train, backtest_period)
            start_test = add_days_to_string_time(start_test, backtest_period)
            end_test = add_days_to_string_time(end_test, backtest_period)
            start_backtest = add_days_to_string_time(start_backtest, backtest_period)
            end_backtest = add_days_to_string_time(end_backtest, backtest_period)
        else:
            days_left = days_between(end_backtest, input_end_backtest)
            start_train = add_days_to_string_time(start_train, days_left)
            end_train = add_days_to_string_time(end_train, days_left)
            start_test = add_days_to_string_time(start_test, days_left)
            end_test = add_days_to_string_time(end_test, days_left)
            start_backtest = add_days_to_string_time(start_backtest, days_left)
            end_backtest = add_days_to_string_time(end_backtest, days_left)
            flag_break = True
            if count == 1:
                break