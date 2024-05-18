#		#		in The Name of God #	#
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from utils.tools import StandardScaler
from utils.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    def __init__(self, root_path, data_path, flag, size, features, target, scale, inverse, timeenc, freq, dtype, take_data_instead_of_reading, direct_data, target_data, cols, kind_of_scaler, scale_with_a_copy_of_target):
        super(Dataset_Custom, self).__init__()

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.size = size
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.dtype = dtype
        self.take_data_instead_of_reading = take_data_instead_of_reading
        self.direct_data = direct_data
        self.target_data = target_data
        self.cols = cols
        self.kind_of_scaler = kind_of_scaler
        self.scale_with_a_copy_of_target = scale_with_a_copy_of_target

        if self.take_data_instead_of_reading:
            self.data = self.data_path
            self.data = np.array(self.data)

            self.origin_data = self.data
        else:
            self._read_data()
        self._gen_data()

    def _read_data(self):
        data_path = self.root_path + self.data_path + self.flag + '.npy'
        data = np.load(data_path)
        self.data = data

    def _gen_data(self):
        # data process
        scale_range = (-1, 1) if self.scale else (0, 1)

        # temporal encoding
        timeenc = np.tile(np.array(range(self.size[0]))[:, None], [1, self.size[1]])
        timeenc = timeenc / max(timeenc.flatten())

        data = self.data

        if self.features == 'M' or self.features == 'S':
            # time features
            if self.timeenc == 1:
                data_time = np.concatenate([np.sin(2 * np.pi * timeenc), np.cos(2 * np.pi * timeenc)], axis=-1)
                data = np.concatenate([data, data_time], axis=-1)
        elif self.features == 'MS':
            # time features
            if self.timeenc == 1:
                data_time = np.concatenate([np.sin(2 * np.pi * timeenc), np.cos(2 * np.pi * timeenc)], axis=-1)
                data = np.concatenate([data, data_time], axis=-1)

        self.data = data

        # target
        self.target = self.data[:, :, self.cols.index(self.target)].reshape((-1, self.size[0], 1))

        # features
        self.features = np.delete(self.data, self.cols.index(self.target), axis=-1)

        # normalize
        if self.scale:
            self.origin_target = self.target.copy()
            self.origin_data = self.data.copy()

            self.target, self.target_scaler = self._scale(self.target)
            self.features, self.feature_scaler = self._scale(self.features)

        if self.inverse:
            self.origin_target = self.target.copy()
            self.origin_data = self.data.copy()

    def _scale(self, data):
        if self.kind_of_scaler == 'MinMax':
            scaler = MinMaxScaler(feature_range=self.scale_range)
        elif self.kind_of_scaler == 'Standard':
            scaler = StandardScaler()
        elif self.kind_of_scaler == 'Robust':
            scaler = RobustScaler()
        elif self.kind_of_scaler == 'Quantile':
            scaler = QuantileTransformer()
        elif self.kind_of_scaler == 'MaxAbs':
            scaler = MaxAbsScaler()
        elif self.kind_of_scaler == 'Power':
            scaler = PowerTransformer()
        else:
            scaler = MinMaxScaler(feature_range=self.scale_range)
        if self.scale_with_a_copy_of_target:
            data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1, self.size[0], 1)
        else:
            data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(-1, self.size[0], data.shape[-1])
        return data, scaler

    def __len__(self):
        return len(self.target) - self.size[0] - self.size[1] + 1

    def __getitem__(self, index):
        i = index
        seq_x = self.features[i:i+self.size[0]].astype(self.dtype)
        seq_y = self.target[i+self.size[0]:i+self.size[0]+self.size[1]].astype(self.dtype)

        seq_x_mark = np.ones([self.size[0], 1]).astype(self.dtype)
        seq_y_mark = np.ones([self.size[1], 1]).astype(self.dtype)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class Dataset_Pred(Dataset):
    def __init__(self, root_path, data_path, flag, size, features, target, scale, inverse, timeenc, freq, dtype, take_data_instead_of_reading, direct_data, target_data, cols, kind_of_scaler, scale_with_a_copy_of_target):
        super(Dataset_Pred, self).__init__()

        self.root_path = root_path
        self.data_path = data_path
        self.flag = flag
        self.size = size
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.dtype = dtype
        self.take_data_instead_of_reading = take_data_instead_of_reading
        self.direct_data = direct_data
        self.target_data = target_data
        self.cols = cols
        self.kind_of_scaler = kind_of_scaler
        self.scale_with_a_copy_of_target = scale_with_a_copy_of_target

        if self.take_data_instead_of_reading:
            self.data = self.data_path
            self.data = np.array(self.data)

            self.origin_data = self.data
        else:
            self._read_data()
        self._gen_data()

    def _read_data(self):
        data_path = self.root_path + self.data_path + self.flag + '.npy'
        data = np.load(data_path)
        self.data = data

    def _gen_data(self):
        # data process
        scale_range = (-1, 1) if self.scale else (0, 1)

        # temporal encoding
        timeenc = np.tile(np.array(range(self.size[0]))[:, None], [1, self.size[1]])
        timeenc = timeenc / max(timeenc.flatten())

        data = self.data

        if self.features == 'M' or self.features == 'S':
            # time features
            if self.timeenc == 1:
                data_time = np.concatenate([np.sin(2 * np.pi * timeenc), np.cos(2 * np.pi * timeenc)], axis=-1)
                data = np.concatenate([data, data_time], axis=-1)
        elif self.features == 'MS':
            # time features
            if self.timeenc == 1:
                data_time = np.concatenate([np.sin(2 * np.pi * timeenc), np.cos(2 * np.pi * timeenc)], axis=-1)
                data = np.concatenate([data, data_time], axis=-1)

        self.data = data

        # target
        self.target = self.data[:, :, self.cols.index(self.target)].reshape((-1, self.size[0], 1))

        # features
        self.features = np.delete(self.data, self.cols.index(self.target), axis=-1)

        # normalize
        if self.scale:
            self.origin_target = self.target.copy()
            self.origin_data = self.data.copy()

            self.target, self.target_scaler = self._scale(self.target)
            self.features, self.feature_scaler = self._scale(self.features)

        if self.inverse:
            self.origin_target = self.target.copy()
            self.origin_data = self.data.copy()

    def _scale(self, data):
        if self.kind_of_scaler == 'MinMax':
            scaler = MinMaxScaler(feature_range=self.scale_range)
        elif self.kind_of_scaler == 'Standard':
            scaler = StandardScaler()
        elif self.kind_of_scaler == 'Robust':
            scaler = RobustScaler()
        elif self.kind_of_scaler == 'Quantile':
            scaler = QuantileTransformer()
        elif self.kind_of_scaler == 'MaxAbs':
            scaler = MaxAbsScaler()
        elif self.kind_of_scaler == 'Power':
            scaler = PowerTransformer()
        else:
            scaler = MinMaxScaler(feature_range=self.scale_range)
        if self.scale_with_a_copy_of_target:
            data = scaler.fit_transform(data.reshape(-1, 1)).reshape(-1, self.size[0], 1)
        else:
            data = scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(-1, self.size[0], data.shape[-1])
        return data, scaler

    def __len__(self):
        return len(self.target) - self.size[0] - self.size[1] + 1

    def __getitem__(self, index):
        i = index
        seq_x = self.features[i:i+self.size[0]].astype(self.dtype)
        seq_y = self.target[i+self.size[0]:i+self.size[0]+self.size[1]].astype(self.dtype)

        seq_x_mark = np.ones([self.size[0], 1]).astype(self.dtype)
        seq_y_mark = np.ones([self.size[1], 1]).astype(self.dtype)

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
