#		#		in The Name of God #	#
import os
import time
import numpy as np
import pandas as pd
import joblib
import warnings
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler, QuantileTransformer, MaxAbsScaler, Normalizer, Binarizer

warnings.filterwarnings('ignore')


def __detecet_scaler__(name):
    if name == 'MinMax':
        scaler = MinMaxScaler()
    elif name == 'Standard':
        scaler = StandardScaler()
    elif name == 'Robust':
        scaler = RobustScaler()
    elif name == 'Quantile':
        scaler = QuantileTransformer()
    elif name == 'MaxAbs':
        scaler = MaxAbsScaler()
    elif name == 'Power':
        scaler = PowerTransformer()
    elif name == 'Normalizer':
        scaler = Normalizer()
    elif name == 'Binarizer':
        scaler = Binarizer()
    else:
        scaler = MinMaxScaler()
    return scaler


class Dataset_Custom(Dataset):
    
    def __init__(self, 
                root_path,                              # Root path where the dataset is located.
                kind_of_scaler='MinMax',                # Type of scaler to use for data normalization. Default is 'MinMax'.
                flag='train',                           # Flag indicating the purpose of the dataset ('train', 'test', 'pred').
                size=None,                              # Size of the dataset: [seq_len, label_len, pred_len].
                test_size=0.2,
                take_data_instead_of_reading=False,     # If True, use direct data instead of reading from a file.
                direct_data_df=None,                       # Direct data to use if `take_data_instead_of_reading` is True.
                features='MS',                          # Features to include in the dataset ('MS', 'M', 'S').
                data_path='data.csv',                   # Path to the dataset file.
                scale_with_a_copy_of_target=False,      # Make a copy of target, and use the copy for y . so it will not take effected by scalling.
                direct_target_column =None,                       # Target data to use if not included in the main data.
                direct_date_column = None,
                direct_to_replace = False,  # There is two side for this :
                #           if direct_target_column is not None and your data has the column with same as name the self.target name , True : replace it with the current column target in data . False: will not change the target in data and will use direct target for -> y -< so it will not effected by scalling
                #           if direct_date_column is not None and and your data has no date column it will place it there. in other wise, it replace it with current one.
                target='Close',                         # Target feature to predict.
                scale=False,                            # If True, scale the data.
                inverse=False,                          # If True, inverse scaling is applied.
                timeenc=0,                              # Type of time encoding to use (0: month, day, weekday, hour).
                freq='b',                               # Frequency of time data ('b': business day, 'h': hour, 't': minute).
                cols=None,                              # Columns to include in the dataset.
                dtype_=None):                           # Data type of the dataset.
        
        if size is None:
            self.seq_len = 10
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'pred']
        type_map = {'train': 0, 'test': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.timeenc = timeenc
        self.freq = freq
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.scale = scale
        self.kind_of_scaler = kind_of_scaler
        self.inverse = inverse
        self.dtype_ = dtype_
        self.scale_with_a_copy_of_target = scale_with_a_copy_of_target
        self.cols = None if cols is None else cols 
        self.target = target
        self.direct_target_column = direct_target_column
        self.direct_date_column = direct_date_column
        self.root_path = root_path
        self.data_path = data_path
        self.direct_to_replace = direct_to_replace
        self.direct_data = direct_data_df
        self.take_data_instead_of_reading = take_data_instead_of_reading
        self.scaler_archive = None
        
        self.__read_data__()
    
    
    def __scale_data__(self, X, id_=None, range_of_fit=None):
        
        if self.inverse:
            if id_ is not None:
                dum_num = np.random.randint(10000)
                random_path = os.path.join(self.root_path, 'scalers', f'dummy{dum_num}prossess')
                file_name = f'{self.kind_of_scaler}Scaler{id_}.joblib'
                try:
                    os.mkdir(random_path)
                except FileExistsError:
                    pass
                except Exception as e:
                    random_path = self.root_path
        try:
            scaler = __detecet_scaler__(self.kind_of_scaler)
        except:
            warnings.warn("the scale job was failed! check the  self.scale  and  self.kind_of_scaler  and make sure the right values of them ! ")
            return X
        range_of_fit = range_of_fit if range_of_fit is not None else ((X.shape[0]) // 2, X.shape[0])
        X_scaler = scaler.fit(X[range_of_fit[0]:range_of_fit[1],])
        if self.inverse and id_ is not None:
            joblib.dump(X_scaler, os.path.join(random_path, file_name))
            self.path_to_scaler_of_target = os.path.join(random_path, file_name)
            #print(f" The Scaler of Target DataColumn {self.target} with id -> {id_} was seccessfuly saved in : ", self.path_to_scaler_of_target)
            time.sleep(1)
            scalled_X = X_scaler.transform(X)
            return scalled_X, random_path, file_name
        else:
            scalled_X = X_scaler.transform(X)
            
            return scalled_X
    
    def __read_data__(self):
        
        if self.take_data_instead_of_reading:
            self.__process_direct_data__()
        else:
            self.__process_from_file__()
    
    def __process_direct_data__(self):
        
        if self.direct_data is None:
            try:
                self.__process_from_file__()
            except:
                print("When 'take_data_instead_of_reading' is True, you should provide 'direct_data'. Exiting.")
                raise
        df_raw = self.direct_data.copy()
        if self.direct_target_column is not None:
            df_raw = self.__update_column_in_data__(df_raw, 'target')
        if self.direct_date_column is not None:
            df_raw = self.__update_column_in_data__(df_raw, 'date')
        
        self.__handle_cols__(df_raw)
    
    def __process_from_file__(self):
        
        try:
            df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), dtype=self.dtype_)
            if self.take_data_instead_of_reading:
                warnings.warn(f"No direct_data found! Instead we read the file inside the {self.root_path}")
            else:
                pass
        except Exception as e:
            print(e)
            raise
        self.__handle_cols__(df_raw)
    
    def __update_column_in_data__(self, df_raw, which_col):
        
        if which_col == 'target':
            if self.target not in df_raw.columns:
                df_raw[self.target] = self.direct_target_column
            else:
                if self.direct_to_replace == False:
                    return df_raw
                else:
                    warnings.warn("Target detected in data! Dropping...")
                    df_raw = df_raw.drop([self.target], axis=1)
                    warnings.warn("Target dropped!")
                    return df_raw
        
        elif which_col == 'date':
            cols = list(df_raw.columns)
            if self.direct_date_column is not None:
                if self.direct_to_replace:
                    df_raw.drop([cols[0]], axis = 1)
                    df_raw['date'] = self.direct_date_column
                    return df_raw
                else:
                    df_raw['date'] = self.direct_date_column
                    return df_raw
            else:
                print("There was no date column detected from your data .!.")
                time.sleep(1)
                try:
                    df_raw = self.__handle_date_column_issue__(df_raw, cols)
                except Exception as e:
                    print(e)
                    raise
        else:
            raise NotImplementedError("NotImplementedError")
    
    def __handle_cols__(self, df_raw):
        
        if self.cols:
            cols = self.cols.copy()
        else:
            cols = list(df_raw.columns)
            self.cols = cols.copy()
        if self.target in cols:
            cols.remove(self.target)
        else:
            if self.direct_target_column is None:
                print("If target is not in main data, pass it manually to 'direct_target_column'!")
                raise ValueError
            else:
                
                df_raw = self.__update_column_in_data__(df_raw,  'target')
        
        self.__arrange_columns__(df_raw)
    
    def __arrange_columns__(self, df_raw):
        
        cols = df_raw.columns
        if 'date' not in cols:
            df_raw = self.__update_column_in_data__(df_raw, 'date')
        try:
            cols.remove('date')
            df_raw[['date'] + cols + [self.target]]
            self.__data_jobs__(df_raw)
        except Exception as e:
            print(f"There was an unexpected issue with the date column.-> {e}")
            raise
    
    def __handle_date_column_issue__(self, df_raw):
        cols = df_raw.columns
        date_name = input("Please enter the correct name of time column  or type q to exit :   ")
        if date_name.lower() == 'q':
            raise
        else:
            try:
                cols.remove(date_name)
                df_raw = df_raw[[date_name] + cols + [self.target]]
                cols.insert(0, 'date')
                cols.append(self.target)
                df_raw.set_axis(cols, axis=1)
                return df_raw
            except Exception as e:
                print(e)
                raise
    
    
    def __data_jobs__(self, df_raw):
        
        target = None
        
        num_train = int(len(df_raw) * self.train_size)
        num_test = int(len(df_raw) * self.test_size)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.scale_with_a_copy_of_target:
            if self.direct_target_column :
                target = self.direct_target_column.copy()
                if isinstance(target, np.ndarray):
                    pass
                else:
                    try:
                        target = np.asarray(target)
                    except:
                        try:
                            target = np.array(target)
                        except:
                            raise ValueError(" You Should Provide direct_target_column is type numpy.ndarray ! ")
                if target.ndim == 1 :
                        target = target.reshape((-1,1))
                elif target.ndim == 2 :
                        pass
                else:
                        raise ValueError(" You Should Provide direct_target_column is 2 dim ! ")
            else:
                target = df_raw[[self.target]].values
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        #scale_map = []
        if self.scale:
            train_data = (border1s[0],border2s[0])
            if self.features == 'M' or self.features == 'MS':
                columns_scaled = []
                for any_ in df_data.columns:
                    temp_col = df_data[[any_]].values
                    if any_ == self.target and self.inverse:
                        col_scaled, path_to_scaler, scaler_name = self.__scale_data__(X=temp_col, id_=any_, range_of_fit = train_data)
                        map_of_scaled_x = {
                                "Before Standard": list(temp_col),
                                "After Standard": list(col_scaled)
                            }
                        self.scaler_archive = {
                            "Scaler Object Path": path_to_scaler,
                            "Scaler Object File Name": scaler_name
                        }
                        #scale_map.append(map_of_scaled_x)
                        map_of_scaled_x = pd.DataFrame(map_of_scaled_x)
                        map_of_scaled_x.to_csv(os.path.join(path_to_scaler, 'Scale Map.csv'), index = False)
                    else:
                        col_scaled = self.__scale_data__(X=temp_col, range_of_fit=train_data)
                    columns_scaled.append(col_scaled)
                scaled_data = np.concatenate(columns_scaled, axis=1)
            else:
                temp_col = df_data.values
                if self.inverse:
                    scaled_data, path_to_scaler, scaler_name = self.__scale_data__(X=temp_col, id_=self.target, range_of_fit=train_data)
                    scale_map = {
                            "Before Standard": list(temp_col),
                            "After Standard": list(scaled_data)
                        }
                    self.scaler_archive = {
                        "Scaler Object Root Path": path_to_scaler,
                        "Scaler Object File Name": scaler_name
                    }
                    scale_map = pd.DataFrame(scale_map)
                    scale_map.to_csv(os.path.join(path_to_scaler, 'Scale Map.csv'), index = False)
                else:
                    scaled_data = self.__scale_data__(temp_col, train_data)
            data = scaled_data
        else:
            data = df_data.values
        
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if target is not None:
            data = data[:, :-1]
            data = np.column_stack((data, target))
        
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    
    def __getitem__(self, index):
        
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        if self.scale_with_a_copy_of_target:
            return data
        else:
            scaler = joblib.load(self.path_to_scaler_of_target)
            return scaler.inverse_transform(data)