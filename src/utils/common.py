import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
import torch.backends.cudnn as cudnn
import random
import importlib
import yaml

data_baseline = np.load("data/US/sp500/baseline_data_sp500.npy", allow_pickle=True).item()
def update_config_to_yaml(source_config, target_config):
    with open(target_config, 'w') as file:
        yaml.dump(source_config, file)

def format_str_datetime(str_date):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return date.strftime('%Y-%m-%d')

def add_days_to_string_time(str_date, days=1):
    date = str(pd.to_datetime(str_date))
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    new_date = date + timedelta(days=days)
    return new_date.strftime('%Y-%m-%d')

def days_between(start_time, end_time):
    d1 = str(pd.to_datetime(start_time))
    d2 = str(pd.to_datetime(end_time))
    d1 = datetime.strptime(d1, "%Y-%m-%d %H:%M:%S")
    d2 = datetime.strptime(d2, "%Y-%m-%d %H:%M:%S")
    return (d2-d1).days

def get_data_baseline(symbol, start_date, end_date, lookback=0, lookforward=0):
    df = data_baseline[symbol]
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_data = df.loc[mask]
    if lookback:
        concat_df = df.loc[:df_data.index[0]].tail(lookback+1)
        df_data = pd.concat([df_data, concat_df], axis=0).sort_index().drop_duplicates()
    if lookforward:
        concat_df = df.loc[df_data.index[-1]:].head(lookforward+1)
        df_data = pd.concat([df_data, concat_df], axis=0).sort_index().drop_duplicates()
    df_data = df_data.astype(float)
    return df_data
def seed(number):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(number)
    np.random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)

def get_class(path):
    path = path.replace('/', '.')
    class_name = path.split(".")[-1]
    module_name = path[0:-len(class_name)-1]
    module_model = importlib.import_module(module_name)
    class_model = getattr(module_model, class_name)
    return class_model