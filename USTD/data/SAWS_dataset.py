# Author: Genevieve Chikwanha
# Date: 10 August 2026
from data.base_dataset import BaseDataset
import torch
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler

from data.data_util import calculate_normalized_laplacian


class SAWSDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(y_dim=1, covariate_dim=4, spatial_dim=64)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.target_channel = 0
        self.opt = opt
        self.time_division = {'train': [0.0, 0.6], 'val': [0.6, 0.8],'test': [0.8, 1.0]}
        self.A = np.load('dataset/saws/adj.npy')
        self.raw_data = self.load_feature('dataset/saws/flow.npy', self.time_division[opt.phase], add_time_in_day=True, add_time_of_year=True, add_height=True)
        # get data division index
        self.opt.__dict__.update({'num_nodes': self.A.shape[0]})
        self.test_node_index = self.get_node_division("", num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)  # all stations used

        # data format check
        self._data_format_check()

    def load_feature(self, data_path, time_division, add_time_in_day=True, add_time_of_year=True, add_height=True):
        flow = np.load(data_path).astype(np.float32) # T, V, D
        X = np.transpose(flow, (1, 0, 2)).copy() # V, T, D
        num_nodes, num_time, num_channels = X.shape
        X = X[:, :, self.target_channel:self.target_channel + 1] # V, T, 1
        self.opt.__dict__.update({'y_dim': 1})

        train_start = int(self.time_division['train'][0] * num_time)
        train_end = int(self.time_division['train'][1] * num_time)
        X_train_slice = X[:, train_start:train_end, :]
        X_mean = np.mean(X_train_slice)
        print("MEAN", X_mean)
        X_std = np.std(X_train_slice)
        print("STD", X_std)
        if np.ndim(X_std) == 0:
            X_std = np.float32(1.0) if X_std == 0 else X_std
        self.add_norm_info(X_mean, X_std)
        X = (X - self.opt.mean) / self.opt.scale
        missing = np.zeros(X.shape)
        print(missing.shape)
        start_time = np.datetime64('2016-01-01T01:00:00')
        full_timestamps = start_time + np.arange(num_time, dtype='int64') * np.timedelta64(1, 'h')
        full_time_seconds = ((full_timestamps - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64)

        feature_list = []
        if add_time_in_day:
            day_floor = full_timestamps.astype('datetime64[D]')
            time_ind = (full_timestamps - day_floor) / np.timedelta64(1, 'D')  # fraction in [0,1), shape (T,)
            time_in_day = np.tile(time_ind[np.newaxis, :, np.newaxis], (num_nodes, 1, 1))  # V, T, 1
            feature_list.append(time_in_day)
        if add_time_of_year:
            day_of_year = pd.DatetimeIndex(full_timestamps).dayofyear.to_numpy().astype(np.float32)
            angle = 2 * np.pi * day_of_year / 365.25
            doy_sin = np.tile(np.sin(angle)[np.newaxis, :, np.newaxis], (num_nodes, 1, 1))  # V, T, 1
            doy_cos = np.tile(np.cos(angle)[np.newaxis, :, np.newaxis], (num_nodes, 1, 1))  # V, T, 1
            feature_list.append(doy_sin)
            feature_list.append(doy_cos)
        if add_height:
            station_heights = np.load('dataset/saws/station_heights.npy').astype(np.float32)  # (V,), meters
            assert station_heights.shape[0] == num_nodes, \
                    f"station_heights has {station_heights.shape[0]} entries, expected {num_nodes} to match adj/flow node order"
            # z-score the heights using train-period stats for consistency with X's normalization approach
            h_mean, h_std = np.mean(station_heights), np.std(station_heights)
            h_std = np.float32(1.0) if h_std == 0 else h_std
            heights_norm = (station_heights - h_mean) / h_std
            height_feat = np.tile(heights_norm[:, np.newaxis, np.newaxis], (1, num_time, 1))  # V, T, 1
            feature_list.append(height_feat)

        if feature_list:
            feat = np.concatenate(feature_list, axis=-1).astype(np.float32)  # V, T, C
        else:
            feat = np.zeros((num_nodes, num_time, 0), dtype=np.float32)

        start_index = int(time_division[0] * num_time)
        end_index = int(time_division[1] * num_time)
        X = X[:, start_index:end_index, :]
        feat = feat[:, start_index:end_index, :]
        missing = missing[:, start_index:end_index, :]
        time_list = full_time_seconds[start_index:end_index]

        return {'pred': X, 'missing': missing, 'time': time_list, 'feat': feat}
