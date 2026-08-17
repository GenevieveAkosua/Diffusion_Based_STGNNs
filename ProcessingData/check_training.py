import numpy as np

flow = np.load('flow.npy')  # (T, V, D)
feature_idx = 0  # whichever single feature AGCRN is trained on

test_start = int(87672 * 0.8)
target = flow[test_start:, :, feature_idx]

# Persistence baseline: predict value at t as value at t-horizon
horizon = 24
y_true = target[horizon:]
y_pred_persistence = target[:-horizon]
mae_persistence = np.mean(np.abs(y_true - y_pred_persistence))
print(f"Persistence MAE: {mae_persistence:.4f}")

# Climatology baseline: predict the mean of the train split
train_mean = flow[:test_start, :, feature_idx].mean()
mae_climatology = np.mean(np.abs(target - train_mean))
print(f"Climatology MAE: {mae_climatology:.4f}")
