import numpy as np
import torch

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

"""
class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min = 0.
        self.max = 1.
        self.feature_range = feature_range
    
    def fit(self, data):
        self.min = data.min(0)
        self.max = data.max(0)
    
    def transform(self, data):
        min_val = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_val = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        scale = (self.feature_range[1] - self.feature_range[0]) / (max_val - min_val)
        return self.feature_range[0] + (data - min_val) * scale

    def inverse_transform(self, data):
        min_val = torch.from_numpy(self.min).type_as(data).to(data.device) if torch.is_tensor(data) else self.min
        max_val = torch.from_numpy(self.max).type_as(data).to(data.device) if torch.is_tensor(data) else self.max
        scale = (max_val - min_val) / (self.feature_range[1] - self.feature_range[0])
        if data.shape[-1] != min_val.shape[-1]:
            min_val = min_val[-1:]
            max_val = max_val[-1:]
            scale = scale[-1:]
        return min_val + (data - self.feature_range[0]) * scale


class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
"""
