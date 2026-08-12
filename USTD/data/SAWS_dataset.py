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
        parser.set_defaults(y_dim=1, covariate_dim=0, spatial_dim=64)
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.target_channel = 0
        self.opt = opt
        self.time_division = {'train': [0.0, 0.6], 'val': [0.6, 0.8],'test': [0.8, 1.0]}
        self.A = np.load('dataset/saws/adj.npy')
        self.raw_data = self.load_feature('dataset/saws/flow.npy', self.time_division[opt.phase])
        # get data division index
        self.opt.__dict__.update({'num_nodes': self.A.shape[0]})
        self.test_node_index = np.array([], dtype=np.int64)          # no nodes held out spatially
        self.train_node_index = np.arange(self.raw_data['pred'].shape[0])  # all stations used

        # data format check
        self._data_format_check()

    def load_feature(self, data_path, time_division):
        flow = np.load(data_path).astype(np.float32) # T, V, D
        X = np.transpose(flow, (1, 0, 2)).copy() # V, T, D
        num_nodes, num_time, num_channels = X.shape
        X = X[:, :, self.target_channel:self.target_channel + 1] # V, T, 1
        self.opt.__dict__.update({'y_dim': 1})

        train_start = int(self.time_division['train'][0] * num_time)
        train_end = int(self.time_division['train'][1] * num_time)
        X_train_slice = X[:, train_start:train_end, :]
        channel_mean = np.mean(X_train_slice, axis=(0, 1)).astype(np.float32)
        channel_std = np.std(X_train_slice, axis=(0, 1)).astype(np.float32)
        channel_std[channel_std == 0] = 1.0
        self.add_norm_info(channel_mean, channel_std)
        X = (X - self.opt.mean) / self.opt.scale
        missing = np.zeros_like(X, dtype=np.float32)
        start_time = np.datetime64('2016-01-01T01:00:00')
        full_timestamps = start_time + np.arange(num_time, dtype='int64') * np.timedelta64(1, 'h')
        full_time_seconds = ((full_timestamps - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64)

        start_index = int(time_division[0] * num_time)
        end_index = int(time_division[1] * num_time)
        X = X[:, start_index:end_index, :]
        missing = missing[:, start_index:end_index, :]
        time_list = full_time_seconds[start_index:end_index]

        return {'pred': X.astype(np.float32), 'missing': missing.astype(np.float32), 'time': time_list}
