# -*- coding: utf-8 -*-
# Adapted by: Genevieve Chikwanha
# Date: 25 July 2026
import numpy as np
import pandas as pd
from timeit import default_timer as timer


def mask_np(array, null_val):
    # If the null value is NaN, create a binary (1/0) arr, wheras if its some number (like 0), find all elements that aren't that number
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

# The null values are zeroed out while the non-nulls are weighted up when getting the metrics

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))

def masked_rmse_np(y_true, y_pred, null_val=np.nan):
    return masked_mse_np(y_true, y_pred, null_val) ** 0.5

def masked_nrmse_np(y_true, y_pred, null_val=np.nan, epsilon=1e-7):
    rmse = masked_rmse_np(y_true, y_pred, null_val)
    valid_mask_bool = mask_np(y_true, null_val).astype(bool)
    std = np.nanstd(y_true[valid_mask_bool])
    return rmse / (std + epsilon)

def masked_vpt_np(y_true, y_pred, null_val=np.nan, threshold=0.5):
    """
    y_true, y_pred: (B, T_p, V, D)
    Returns dict of VPT summary stats over the batch, analogous to Henok's vpt_batch.
    """
    B, T_p = y_true.shape[0], y_true.shape[1]
    vpts = np.full(B, T_p, dtype=float)

    for b in range(B):
        for t in range(T_p):
            step_nrmse = masked_nrmse_np(y_true[b:b+1, t], y_pred[b:b+1, t], null_val)
            if step_nrmse > threshold:
                vpts[b] = t
                break

    return {'vpt_mean': float(np.mean(vpts)), 'vpt_std': float(np.std(vpts)), 'vpt_median': float(np.median(vpts)), 'vpt_min': float(np.min(vpts)), 'vpt_max': float(np.max(vpts)), 'vpt_dist': vpts}

def time_to_str(t, mode='min'):
    """Formats time t as hours & mins OR mins & secs from seconds"""

    if mode=='min':
        t = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError

class Metric(object):
    """Computes and stores the average and current value,调用时先reset"""

    def __init__(self, T_p):
        self.time_start = timer()
        self.T_p = T_p
        self.best_metrics = {'mae': np.inf, 'mse': np.inf, 'rmse': np.inf, 'nrmse': np.inf, 'mape': np.inf, 'crps':np.inf, 'mis': -np.inf, 'vpt': -np.inf, 'epoch': np.inf}
        self.metrics = {}
        self.vpt_threshold = 0.5
        # self.step_metrics_epoch = {'mae': {}, 'rmse': {}, 'mape': {}}

    def update_metrics(self, y_true, y_pred, eval_mask=None):
        """
        both y_true and y_pred should be numpy
        :param y_true: (B, T_p, V, D)
        :param y_pred: (B, n_samples, T_p, V, D) or (B, T_p, V, D)
        :return:
        """
        assert  isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), \
            f"y_true and y_pred should be np.ndarray, now its type is y_true:{type(y_true)}, y_pred:{type(y_pred)}"
        self.metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mape': 0.0, 'crps': 0, 'mis': 0.0, 'vpt': 0, 'time': 0.0}

        if y_pred.shape == y_true.shape: #  y_true: (B, T_p, V, D) and y_pred: (B, T_p, V, D)
            y_pred = np.expand_dims(y_pred, axis = 1) # (B, 1, T_p, V, D)

        # probabilistic metric
        eval_points = np.ones_like(y_true)
        self.metrics['crps'] = calc_quantile_CRPS(torch.from_numpy(y_true), torch.from_numpy(y_pred), torch.from_numpy(eval_points))
        self.metrics['mis'] = calc_mis(torch.from_numpy(y_true), torch.from_numpy(y_pred))

        # deterministic metric
        y_pred = np.mean(y_pred, axis=1) # # (B, T_p, V, D)

        self.metrics['mae'], self.metrics['rmse'], self.metrics['mape'], self.metrics['mse'], self.metrics['vpt'] = self.get_metric(y_true, y_pred)
        # self.metrics['time'] = time_to_str((timer() - self.time_start))
        self.metrics['time'] = timer() - self.time_start

    def update_best_metrics(self, epoch=0):
        self.best_metrics['mae'], mae_state = self.get_best_metric(self.best_metrics['mae'], self.metrics['mae'])
        self.best_metrics['rmse'], rmse_state = self.get_best_metric(self.best_metrics['rmse'], self.metrics['rmse'])
        self.best_metrics['mape'], mape_state = self.get_best_metric(self.best_metrics['mape'], self.metrics['mape'])
        self.best_metrics['crps'], crps_state = self.get_best_metric(self.best_metrics['crps'], self.metrics['crps'])
        self.best_metrics['mis'], mis_state = self.get_best_metric(self.best_metrics['mis'], self.metrics['mis'])
        self.best_metrics['vpt'], _ = self.get_best_metric(self.best_metrics['vpt'], self.metrics['vpt'], higher_best=True)

        if mae_state:
            self.best_metrics['epoch'] = int(epoch)

    @staticmethod
    def get_metric(y_true, y_pred):
        mae = masked_mae_np(y_true, y_pred, np.nan)
        mse = masked_mse_np(y_true, y_pred, np.nan)
        mape = masked_mape_np(y_true, y_pred, np.nan)
        rmse = mse ** 0.5
        vpt = masked_vpt_np(y_true, y_pred, np.nan, 0.5)['vpt_mean']
        return mae, rmse, mape, mse, vpt

    @staticmethod
    def get_best_metric(best, candidate, higher_best=False):
        state = False
        if (candidate > best) if higher_best else (candidate < best):
            best = candidate
            state = True
        return best, state

    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f}{self.metrics['vpt']:<7.2f}{self.metrics['crps']:<7.2f} {self.metrics['mis']:<7.2f} | {self.best_metrics['epoch'] + 1:<4} "

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f},{self.best_metrics['vpt']},{self.best_metrics['crps']},{self.best_metrics['mis']},{self.best_metrics['epoch']}"

    def log_lst(self, epoch=None, sep=','):
        message_lst = []
        index = ['mae', 'rmse', 'mape', 'vpt', 'crps', 'mis']

        for i in index:
            message_lst.append(f"{i},{self.multi_step_str(obj=i, sep=sep, epoch=epoch)}")
        return message_lst

    def to_dict(self):
        return self.metrics


# metric for Probabilistic evaluation
import torch
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points):
    """
    target: (B, T, V), torch.Tensor
    forecast: (B, n_sample, T, V), torch.Tensor
    eval_points: (B, T, V): which values should be evaluated,
    """

    # target = target * scaler + mean_scaler
    # forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def MIS(target: np.ndarray, lower_quantile: np.ndarray, upper_quantile: np.ndarray, alpha: float) -> float:
    r"""
    mean interval score
    Implementation comes form glounts.evalution metrics
    .. math::
    msis = mean(U - L + 2/alpha * (L-Y) * I[Y<L] + 2/alpha * (Y-U) * I[Y>U])
    """
    numerator = np.mean(
        upper_quantile
        - lower_quantile
        + 2.0 / alpha * (lower_quantile - target) * (target < lower_quantile)
        + 2.0 / alpha * (target - upper_quantile) * (target > upper_quantile)
    )
    return numerator

def calc_mis(target, forecast, alpha = 0.05):
    """
       target: (B, T, V),
       forecast: (B, n_sample, T, V)
    """
    return MIS(target = target.cpu().numpy(), lower_quantile = torch.quantile(forecast, alpha / 2, dim=1).cpu().numpy(), upper_quantile = torch.quantile(forecast, 1.0 - alpha / 2, dim=1).cpu().numpy(), alpha = alpha)


if __name__ == "__main__":
    time_to_str(240, mode='min')

    # test for CRPS
    B, T, V = 32, 12, 36
    n_sample = 100
    target = torch.randn((B, T, V))
    forecast = torch.randn((B, n_sample, T, V))
    label = target.unsqueeze(1).expand_as(forecast)
    eval_points = torch.randn_like(target)

    crps = calc_quantile_CRPS(target, forecast, eval_points)
    print('crps:', crps)

    crps = calc_quantile_CRPS(target, label, eval_points)
    print('crps:', crps)

    mis = calc_mis(target, forecast)
    print('mis:', mis)

    mis = calc_mis(target, label)
    print('mis:', mis)
