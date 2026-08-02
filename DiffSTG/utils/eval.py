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

def masked_smape_np(y_true, y_pred, null_val=np.nan, epsilon=1e-7):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        numerator = np.abs(y_pred - y_true)
        denominator = np.abs(y_pred) + np.abs(y_true) + epsilon
        smape = 200.0 * numerator / denominator
        smape = np.nan_to_num(mask * smape)
        return np.mean(smape)

def masked_vpt_np(y_true, y_pred, null_val=np.nan, threshold=0.5):
    T_p = y_true.shape[1]
    for t in range(T_p):
        step_nrmse = masked_nrmse_np(y_true[:, t], y_pred[:, t], null_val)
        if step_nrmse > threshold:
            return t
    return T_p

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
        self.best_metrics = {'mae': np.inf, 'mse': np.inf, 'rmse': np.inf, 'nrmse': np.inf, 'mape': np.inf, 'smape': np.inf, 'crps':np.inf, 'mis': -np.inf, 'vpt': -np.inf, 'epoch': np.inf}
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
        self.metrics = {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'nrmse': 0.0, 'mape': 0.0, 'smape': 0.0, 'crps': 0, 'mis': 0.0, 'vpt': 0, 'time': 0.0}

        if y_pred.shape == y_true.shape: #  y_true: (B, T_p, V, D) and y_pred: (B, T_p, V, D)
            y_pred = np.expand_dims(y_pred, axis = 1) # (B, 1, T_p, V, D)

        # probabilistic metric
        eval_points = np.ones_like(y_true)
        self.metrics['crps'] = calc_quantile_CRPS(torch.from_numpy(y_true), torch.from_numpy(y_pred), torch.from_numpy(eval_points))
        self.metrics['mis'] = calc_mis(torch.from_numpy(y_true), torch.from_numpy(y_pred))

        # deterministic metric
        y_pred = np.mean(y_pred, axis=1) # # (B, T_p, V, D)

        self.metrics['mae'], self.metrics['rmse'], self.metrics['mape'], self.metrics['mse'], self.metrics['nrmse'], self.metrics['smape'], self.metrics['vpt'] = self.get_metric(y_true, y_pred)
        # self.metrics['time'] = time_to_str((timer() - self.time_start))
        self.metrics['time'] = timer() - self.time_start


    def update_best_metrics(self, epoch=0):
        self.best_metrics['mae'], mae_state = self.get_best_metric(self.best_metrics['mae'], self.metrics['mae'])
        self.best_metrics['rmse'], rmse_state = self.get_best_metric(self.best_metrics['rmse'], self.metrics['rmse'])
        self.best_metrics['mape'], mape_state = self.get_best_metric(self.best_metrics['mape'], self.metrics['mape'])
        self.best_metrics['crps'], crps_state = self.get_best_metric(self.best_metrics['crps'], self.metrics['crps'])
        self.best_metrics['mis'], mis_state = self.get_best_metric(self.best_metrics['mis'], self.metrics['mis'])
        self.best_metrics['nrmse'], _ = self.get_best_metric(self.best_metrics['nrmse'], self.metrics['nrmse'])
        self.best_metrics['smape'], _ = self.get_best_metric(self.best_metrics['smape'], self.metrics['smape'])
        self.best_metrics['vpt'], _ = self.get_best_metric(self.best_metrics['vpt'], self.metrics['vpt'], higher_best=True)

        if mae_state:
            self.best_metrics['epoch'] = int(epoch)

    @staticmethod
    def get_metric(y_true, y_pred):
        mae = masked_mae_np(y_true, y_pred, np.nan)
        mse = masked_mse_np(y_true, y_pred, np.nan)
        mape = masked_mape_np(y_true, y_pred, np.nan)
        rmse = mse ** 0.5
        nrmse = masked_nrmse_np(y_true, y_pred, np.nan, 1e-7)
        smape = masked_smape_np(y_true, y_pred, np.nan, 1e-7)
        vpt = masked_vpt_np(y_true, y_pred, np.nan, 0.5)
        return mae, rmse, mape, mse, nrmse, smape, vpt

    @staticmethod
    def get_best_metric(best, candidate, higher_best=False):
        state = False
        if (candidate > best) if higher_best else (candidate < best):

            best = candidate
            state = True
        return best, state

    def __str__(self):
        """For print"""
        return f"{self.metrics['mae']:<7.2f}{self.metrics['rmse']:<7.2f}{self.metrics['mape']:<7.2f}{self.metrics['smape']:<7.2f}{self.metrics['vpt']:<7.2f}{self.metrics['crps']:<7.2f} {self.metrics['mis']:<7.2f} | {self.best_metrics['epoch'] + 1:<4} "

    def best_str(self):
        """For save"""
        return f"{self.best_metrics['epoch']},{self.best_metrics['mae']:.2f},{self.best_metrics['rmse']:.2f},{self.best_metrics['mape']:.2f},{self.best_metrics['smape']},{self.best_metrics['vpt']},{self.best_metrics['crps']},{self.best_metrics['mis']},{self.best_metrics['epoch']}"


    def log_lst(self, epoch=None, sep=','):
        message_lst = []
        index = ['mae', 'rmse', 'mape', 'smape', 'vpt', 'crps', 'mis']

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

    print("--- Mathematical Verification Test ---")

    # 1. Create a tiny dataset: Shape (1, 1, 3, 1) -> Batch 1, Time 1, Stations 3, Feature 1
    y_true = np.array([[[[10.0], [np.nan], [20.0]]]])
    y_pred = np.array([[[[12.0], [50.0],   [15.0]]]])

    print("Dummy Ground Truth:", y_true.flatten())
    print("Dummy Predictions: ", y_pred.flatten())
    print("-" * 40)

    # 2. Calculate and assert matches
    mae = masked_mae_np(y_true, y_pred, null_val=np.nan)
    mse = masked_mse_np(y_true, y_pred, null_val=np.nan)
    rmse = masked_rmse_np(y_true, y_pred, null_val=np.nan)
    nrmse = masked_nrmse_np(y_true, y_pred, null_val=np.nan)
    mape = masked_mape_np(y_true, y_pred, null_val=np.nan)
    smape = masked_smape_np(y_true, y_pred, null_val=np.nan)

    print(f"{'Metric':<10} | {'Expected':<10} | {'Calculated':<10} | Match?")
    print("-" * 45)
    print(f"{'MAE':<10} | {'3.5':<10} | {mae:<10.4f} | {np.isclose(mae, 3.5)}")
    print(f"{'MSE':<10} | {'14.5':<10} | {mse:<10.4f} | {np.isclose(mse, 14.5)}")
    print(f"{'RMSE':<10} | {'3.8079':<10} | {rmse:<10.4f} | {np.isclose(rmse, 3.807886, atol=1e-4)}")
    print(f"{'NRMSE':<10} | {'0.7616':<10} | {nrmse:<10.4f} | {np.isclose(nrmse, 0.761577, atol=1e-4)}")
    print(f"{'MAPE':<10} | {'22.5':<10} | {mape:<10.4f} | {np.isclose(mape, 22.5)}")
    print(f"{'SMAPE':<10} | {'23.3766':<10} | {smape:<10.4f} | {np.isclose(smape, 23.3766, atol=1e-4)}")

    print("--- VPT Verification Test ---")

    # Shape: (B=1, T_p=3, V=2, D=1)
    # 1 batch, 3 time steps, 2 stations, 1 feature
    y_true_vpt = np.array([
        [[[10.0], [20.0]],   # t=0
         [[10.0], [20.0]],   # t=1
         [[10.0], [20.0]]]   # t=2
    ])

    y_pred_vpt = np.array([
        [[[11.0], [19.0]],   # t=0 -> Errors are 1 and -1. RMSE = 1.
         [[12.0], [18.0]],   # t=1 -> Errors are 2 and -2. RMSE = 2.
         [[15.0], [15.0]]]   # t=2 -> Errors are 5 and -5. RMSE = 5.
    ])

    # Note: Standard deviation of [10, 20] is 5.
    # Therefore, NRMSE = RMSE / 5.
    # t=0 NRMSE: 1 / 5 = 0.2
    # t=1 NRMSE: 2 / 5 = 0.4
    # t=2 NRMSE: 5 / 5 = 1.0 (Threshold breach!)

    threshold = 0.5
    vpt = masked_vpt_np(y_true_vpt, y_pred_vpt, null_val=np.nan, threshold=threshold)

    print(f"Threshold set to: {threshold}\n")
    print(f"{'Time Step (t)':<15} | {'Hand-Calculated NRMSE':<25} | {'Action':<15}")
    print("-" * 60)
    print(f"{'t = 0':<15} | {'1.0 / 5.0 = 0.20':<25} | {'Continue'}")
    print(f"{'t = 1':<15} | {'2.0 / 5.0 = 0.40':<25} | {'Continue'}")
    print(f"{'t = 2':<15} | {'5.0 / 5.0 = 1.00':<25} | {'Breach! Return 2'}")
    print("-" * 60)

    print(f"\nExpected VPT Output: 2")
    print(f"Calculated VPT Output: {vpt}")
    print(f"Match? {vpt == 2}")

    # Simulating your data shapes based on config
    B = 8          # Batch size
    n_samples = 4  # Diffusion model samples
    T_p = 12       # Time steps (e.g., 12 points for 1 hour of 5-min intervals)
    V = 170        # Vertices (Stations/Nodes from the SAWS data)
    F = 6          # Features (e.g., rainfall, humidity, pressure)

    print(f"Expected shapes -> y_true: ({B}, {T_p}, {V}, {F}) | y_pred: ({B}, {n_samples}, {T_p}, {V}, {F})")

    # 1. Generate dummy data simulating your flow.npy structure
    # y_true is the ground truth
    y_true = np.random.rand(B, T_p, V, F).astype(np.float32)

    # Introduce null values to ensure mask_np zeroes them out correctly
    y_true[0, 0, 0, 0] = 0.0
    y_true[0, 0, 0, 1] = np.nan

    # y_pred out of the diffusion model includes the n_samples dimension
    y_pred = np.random.rand(B, n_samples, T_p, V, F).astype(np.float32)

    # 2. Test the Metric class wrapper
    print("\n--- Testing Metric Class Wrapper ---")
    metric = Metric(T_p=T_p)

    # FIX: Manually setting vpt_threshold since it's missing in __init__
    metric.vpt_threshold = 0.5

    try:
        # Note: update_metrics dynamically averages over n_samples for deterministic metrics
        metric.update_metrics(y_true, y_pred)
        print("Metric update successful! Calculated Metrics:")
        for k, v in metric.metrics.items():
            if isinstance(v, float):
                print(f"  {k.upper()}: {v:.4f}")
            else:
                print(f"  {k.upper()}: {v}")
    except Exception as e:
        print(f"Error in Metric update: {e}")

    # 3. Test Individual Masked Functions (Math Verification)
    print("\n--- Testing Individual Masked Metrics ---")
    # For individual deterministic tests, we need y_pred to match y_true shape (mean over samples)
    y_pred_mean = np.mean(y_pred, axis=1)

    mae = masked_mae_np(y_true, y_pred_mean, null_val=0.0)
    mse = masked_mse_np(y_true, y_pred_mean, null_val=0.0)
    rmse = masked_rmse_np(y_true, y_pred_mean, null_val=0.0)
    mape = masked_mape_np(y_true, y_pred_mean, null_val=0.0)

    print(f"Masked MAE:  {mae:.4f}")
    print(f"Masked MSE:  {mse:.4f}")
    print(f"Masked RMSE: {rmse:.4f}")
    print(f"Masked MAPE: {mape:.4f}")
