import numpy as np
import matplotlib.pyplot as plt

# 1. Load the heavy arrays
print("Loading files...")
y_true = np.load('SAWS_true.npy')
y_pred = np.load('SAWS_pred.npy')

# 2. Check the dimensions
# Expected shape is usually: (Batch/Samples, Horizon, Nodes, Features)
print("-" * 30)
print(f"True Data Shape: {y_true.shape}")
print(f"Pred Data Shape: {y_pred.shape}")
print("-" * 30)

# 3. Calculate quick global metrics
mae = np.mean(np.abs(y_pred - y_true))
rmse = np.sqrt(np.mean((y_pred - y_true)**2))

print(f"Global MAE:  {mae:.4f}")
print(f"Global RMSE: {rmse:.4f}")
print("-" * 30)

# 4. Save a simple visual comparison 
# This looks at the very first test sample, the first node, and the first feature 
# across all future time steps (your lag/horizon).
plt.figure(figsize=(10, 5))

# Using index [0, :, 0, 0] -> Sample 0, All Horizons, Node 0, Feature 0
plt.plot(y_true[0, :, 0, 0], label='True Values', marker='o', linestyle='-')
plt.plot(y_pred[0, :, 0, 0], label='Predicted Values', marker='x', linestyle='--')

plt.title('Prediction vs True (Sample 0, Node 0, Feature 0)')
plt.xlabel('Horizon Time Step')
plt.ylabel('Target Value')
plt.legend()
plt.grid(True)

# Saving to file since you are on a remote cluster without a direct GUI
plot_filename = 'saws_quick_check.png'
plt.savefig(plot_filename, bbox_inches='tight')
print(f"Saved a visual comparison to '{plot_filename}'")

sample_idx = 0
node_idx = 0
feature_idx = 0

true_sequence = y_true[sample_idx, :, node_idx, feature_idx]
pred_sequence = y_pred[sample_idx, :, node_idx, feature_idx]

# Print a clean, side-by-side comparison
print(f"--- Sample {sample_idx} | Node {node_idx} | Feature {feature_idx} ---")
print(f"{'Horizon Step':<15} | {'True Value':<15} | {'Predicted Value':<15}")
print("-" * 52)

for step in range(len(true_sequence)):
    t_val = true_sequence[step]
    p_val = pred_sequence[step]
    print(f"Step {step+1:<10} | {t_val:<15.4f} | {p_val:<15.4f}")
