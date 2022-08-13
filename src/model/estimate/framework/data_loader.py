import numpy as np
from utils.common import get_data_baseline
from utils.data_loader import DataLoaderBase
from model.estimate.framework.snapshot import  HypergraphSnapshots
from scipy import sparse
from torch_geometric import utils
import pandas as pd
import os
import torch

class DataLoader(DataLoaderBase):
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
        self._cuda = config["model"]["cuda"]
        self._start_train = config["data"]["start_train"]


    def get_data(self, start_train, end_train, start_test, end_test):
        path_data = "{}/train_{}_{}.npz".format(self._pretrained_log, start_train, end_train)
        if not os.path.exists(path_data):
            X_train_full, y_train_full, X_test_full, y_test_full, hypergraphsnapshot, edges, buy_prob_threshold, sell_prob_threshold = self._split_train_test(start_train, end_train, start_test, end_test)
            np.savez(path_data, x_train=X_train_full, y_train=y_train_full, x_test=X_test_full, y_test=y_test_full, edges=edges, buy_thres=buy_prob_threshold, sell_thres=sell_prob_threshold)
        else:
            data = np.load(path_data)
            X_train_full = data["x_train"]
            y_train_full = data["y_train"]  
            X_test_full = data["x_test"]
            y_test_full = data["y_test"]
            edges = data["edges"]
            buy_prob_threshold = data["buy_thres"]
            sell_prob_threshold = data["sell_thres"]
            train_data_storage = {}
            for sym in self._symbols:
                train_data = get_data_baseline(sym, start_train, end_train, self._his_window + self._max_slow_period, self._n_step_ahead)
                train_data_storage[sym] = train_data
            train_data_storage = self._fill_missing_data_Trung(train_data_storage)
            hypergraphsnapshot = HypergraphSnapshots(self._symbols, self._start_train, train_data_storage, self._cuda)
        return X_train_full, y_train_full, X_test_full, y_test_full, hypergraphsnapshot, torch.LongTensor(edges), buy_prob_threshold, sell_prob_threshold  

    def _split_train_test(self, start_train, end_train, start_test, end_test):
        X_train_storage = []
        X_test_storage = []
        y_train_storage = []
        y_test_storage = []
        train_data_storage = {}
        test_data_storage = {}
        buy_prob_threshold = []
        sell_prob_threshold = []
        print("Downloading data...")
        for sym in self._symbols:
            train_data = get_data_baseline(sym, start_train, end_train, self._his_window + self._max_slow_period, self._n_step_ahead)
            train_data_storage[sym] = train_data
            test_data = get_data_baseline(sym, start_test, end_test, self._his_window + self._max_slow_period, self._n_step_ahead)
            test_data_storage[sym] = test_data
        
        train_data_storage = self._fill_missing_data_Trung(train_data_storage)
        test_data_storage = self._fill_missing_data_Trung(test_data_storage)

        print("Processing data...")
        for sym in train_data_storage:
            _, df_x_train, df_y_train = self._preprocess_data(train_data_storage, sym, self._indicators)
            _, df_x_test, df_y_test = self._preprocess_data(test_data_storage, sym, self._indicators)
            buy_prob_threshold.append(df_y_train.mean())
            sell_prob_threshold.append(-df_y_train.mean())
            train_x = df_x_train.to_numpy()
            train_y = df_y_train.to_numpy()
            test_x = df_x_test.to_numpy()
            test_y = df_y_test.to_numpy()

            X_train = np.array([train_x[i: i + self._his_window]
                                for i in range(len(train_x)-self._his_window)])
            y_train = np.array([train_y[i + 1: i + self._his_window + 1]
                                for i in range(len(train_y)-self._his_window)])
            X_test = np.array([test_x[i: i + self._his_window]
                            for i in range(len(test_x)-self._his_window)])
            y_test = np.array([test_y[i + 1: i + self._his_window + 1]
                            for i in range(len(test_y)-self._his_window)])
            
            X_train_storage.append(X_train)
            X_test_storage.append(X_test)
            y_train_storage.append(y_train)
            y_test_storage.append(y_test)

        X_train_full = np.stack((X_train_storage), axis=1)
        y_train_full = np.stack((y_train_storage), axis = 1)
        X_test_full = np.stack((X_test_storage), axis=1)
        y_test_full =  np.stack((y_test_storage), axis = 1)

        stock_list = pd.read_csv("data/US/sp500/sp500_ticker.csv", index_col = "Symbol")
        cat_list = stock_list.loc[self._symbols]["Sector"].unique()
        cat_dict = {}
        for i in range(len(cat_list)):
            cat = cat_list[i]
            cat_dict[cat] = i
            
        incidence_matrix = np.zeros((len(self._symbols), len(cat_list)))
        for i in range(len(self._symbols)):
            cat_key = stock_list.loc[self._symbols[i]].Sector    
            cat_index = cat_dict[cat_key]
            incidence_matrix[i][cat_index] = 1
            
        inci_sparse = sparse.coo_matrix(incidence_matrix)
        incidence_edges = utils.from_scipy_sparse_matrix(inci_sparse)
        hypergraphsnapshot = HypergraphSnapshots(self._symbols, self._start_train, train_data_storage, self._cuda)
        print("Done!")
        return X_train_full, y_train_full, X_test_full, y_test_full, hypergraphsnapshot, incidence_edges[0], buy_prob_threshold, sell_prob_threshold