#            #    in The Name of GOD  # #
#
# #  These are some simple and straightforward classes  # #
#          to interact with our losses functions          #
# cloner174.org@gmail.com
#
import torch
import torch.nn as nn



class WeightedMeanAbsolutePercentageError(nn.Module):
    
    def __init__(self):
        
        super(WeightedMeanAbsolutePercentageError, self).__init__()
    
    
    def forward(self, y_pred, y_true):
        
        absolute_percentage_errors = torch.abs((y_true - y_pred) / (y_true + 1e-8))
        weighted_errors = absolute_percentage_errors * (torch.abs(y_true) + 1e-8)
        
        return torch.mean(weighted_errors)
    


class SymmetricMeanAbsolutePercentageError(nn.Module):
    
    def __init__(self):
        
        super(SymmetricMeanAbsolutePercentageError, self).__init__()
    
    def forward(self, y_pred, y_true):
        
        absolute_percentage_errors = torch.abs((y_true - y_pred) / ((torch.abs(y_true) + torch.abs(y_pred)) / 2 + 1e-8))
        
        return torch.mean(absolute_percentage_errors)
    


class RMSELoss(nn.Module):
    
    def __init__(self):
        
        super(RMSELoss, self).__init__()
    
    
    def forward(self, y_pred, y_true):
        
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    


class QuantileLoss(nn.Module):
    
    def __init__(self, quantile=0.5):
        
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    
    def forward(self, y_pred, y_true):
        
        errors = y_true - y_pred
        quantile_loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        
        return torch.mean(quantile_loss)
    


class HuberLoss(nn.Module):
    
    def __init__(self, delta=1.0):
        
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    
    def forward(self, y_pred, y_true):
        
        errors = torch.abs(y_pred - y_true)
        quadratic = torch.min(errors, self.delta)
        linear = errors - quadratic
        
        return torch.mean(0.5 * quadratic ** 2 + self.delta * linear)
    


class PinballLoss(nn.Module):
    
    def __init__(self, tau=0.5):
        
        super(PinballLoss, self).__init__()
        self.tau = tau
    
    
    def forward(self, y_pred, y_true):
        
        delta = y_pred - y_true
        loss = torch.max((self.tau - 1) * delta, self.tau * delta)
        
        return torch.mean(loss)
    
#
