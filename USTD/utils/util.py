# Edited by: Genevieve Chikwanha
# Code partly adapted from ChaosBenchNet (Moges, 2026)
# Date: 25 July 2026

"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

#####################################
# evaluation metrics
#####################################
def _rmse_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        rmse: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    rmse = np.sqrt((((y - label) ** 2) * valid_mask).sum() / (valid_count + 1e-7))

    return rmse

def _nrmse_with_missing(y, label, missing_mask):
    rmse = _rmse_with_missing(y, label, missing_mask)
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask * np.ones_like(label))
    mean = (label * valid_mask).sum() / valid_count
    var = (((label - mean) ** 2) * valid_mask).sum() / valid_count
    std = np.sqrt(var)
    return rmse / (std + 1e-7)

    #valid_mask_bool = (1 - missing_mask).astype(bool)
    #valid_labels = label[valid_mask_bool]
    #std = np.std(valid_labels)
    #nrmse = rmse / (std + 1e-7)
    #return nrmse


def _mae_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mae: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    mae = np.abs((y-label) * valid_mask).sum() / valid_count
    return mae

def _mape_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mape: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_mask = valid_mask * (np.abs(label) > 0.0001)
    valid_count = np.sum(valid_mask)

    mape = np.abs((y-label) / (label+1e-6) * valid_mask).sum() / valid_count
    return mape

def _smape_with_missing(y, label, missing_mask):
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    #valid_mask = valid_mask * (np.abs(label) > 0.0001)
    valid_count = np.sum(valid_mask)

    smape = 200.00 * (np.abs(y-label) / (np.abs(label) + np.abs(y) + 1e-6) * valid_mask).sum() / valid_count
    
    #smape = 200.00 * (np.abs(y-label) / (np.abs(label) + np.abs(y) + 1e-6) * valid_mask).sum() / (valid_count + 1e-7)
    return smape

def _quantile_CRPS_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [time, num_sample, num_m, dy]
        label: nd.array [time, num_m, dy]
        missing_index: [time, num_m, 1] or [time, num_m]
    Returns:
        CRPS: float
    """
    y = y.transpose(1, 0, 2, 3) # [num_sample, time, num_m, dy]
    def quantile_loss(target, forecast, q: float, eval_points) -> float:
        return 2 * np.sum(
            np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        )

    def calc_denominator(label, valid_mask):
        return np.sum(np.abs(label * valid_mask))

    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape[:2] == y.shape[:2]:
        missing_mask = missing_mask[:, :, np.newaxis]

    valid_mask = 1 - missing_mask
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(label, valid_mask)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = np.quantile(y, quantiles[i], axis=0)
        q_loss = quantile_loss(label, q_pred, quantiles[i], valid_mask)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def _vpt_with_missing(y, label, missing_mask, threshold=0.5):
    L = label.shape[2]
    for t in range(L):
        step_nrmse = _nrmse_with_missing(y[:, :, t], label[:, :, t], missing_mask[:, :, t])
        if step_nrmse > threshold:
            return t
    return L

if __name__ == "__main__":

    def check(name, expected, actual, atol=1e-4):
        match = np.isclose(expected, actual, atol=atol)
        print(f"{name:<12} | expected: {expected:<10.4f} | actual: {actual:<10.4f} | match: {match}")

    print("=" * 60)
    print("Test 1: MAE / RMSE / MAPE / SMAPE / NRMSE, simple 1D case")
    print("=" * 60)
    # label = [10, 20, 30], missing_mask = [0, 0, 1] -> last point excluded
    # valid pred/label pairs: (12,10) and (18,20); the 999 pred at the masked
    # position should have zero influence on every metric below.
    label1 = np.array([10.0, 20.0, 30.0])
    y1     = np.array([12.0, 18.0, 999.0])
    mask1  = np.array([0.0, 0.0, 1.0])

    check("MAE",   2.0,        _mae_with_missing(y1, label1, mask1))
    check("RMSE",  2.0,        _rmse_with_missing(y1, label1, mask1))
    check("MAPE",  0.15,       _mape_with_missing(y1, label1, mask1))
    check("SMAPE", 14.3541,    _smape_with_missing(y1, label1, mask1))
    check("NRMSE", 0.4,        _nrmse_with_missing(y1, label1, mask1))

    print()
    print("=" * 60)
    print("Test 2: multi-feature (D=3) with a mask that has NO feature dim")
    print("This is the exact shape that crashed the old boolean-index version")
    print("=" * 60)
    # 2 nodes, 3 features each. Node 0 fully valid, node 1 fully missing.
    label2 = np.array([[10.0, 20.0, 30.0],
                        [100.0, 200.0, 300.0]])
    y2     = np.array([[12.0, 18.0, 25.0],
                        [90.0, 210.0, 290.0]])
    mask2  = np.array([0.0, 1.0])   # shape (2,) -- no trailing feature dim

    try:
        nrmse2 = _nrmse_with_missing(y2, label2, mask2)
        check("NRMSE", 0.7036, nrmse2)
        print("-> no crash: fix confirmed working on a D>1 mask.")
    except IndexError as e:
        print(f"-> STILL CRASHING: {e}")
        print("-> _nrmse_with_missing still uses boolean indexing; re-apply the fix.")

    print()
    print("=" * 60)
    print("Test 3: VPT (Valid Prediction Time)")
    print("=" * 60)
    # shape (B=1, N=2, L=3, D=1). Two stations at 10 and 20, so std=5 at every t.
    # Prediction error grows each timestep: RMSE = 1, 2, 5 -> NRMSE = 0.2, 0.4, 1.0
    # threshold=0.5 -> should breach and return t=2
    label3 = np.array([[[[10.0], [10.0], [10.0]],
                         [[20.0], [20.0], [20.0]]]])
    y3     = np.array([[[[11.0], [12.0], [15.0]],
                         [[19.0], [18.0], [15.0]]]])
    mask3  = np.zeros((1, 2, 3))

    for t in range(3):
        step_nrmse = _nrmse_with_missing(y3[:, :, t], label3[:, :, t], mask3[:, :, t])
        print(f"  t={t} NRMSE={step_nrmse:.4f}")

    vpt3 = _vpt_with_missing(y3, label3, mask3, threshold=0.5)
    print(f"VPT expected: 2 | actual: {vpt3} | match: {vpt3 == 2}")
