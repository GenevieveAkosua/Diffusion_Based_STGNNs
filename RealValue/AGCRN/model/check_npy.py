import pandas as pd
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

node_idx = 0
feature_idx = 0
horizon = 48
num_blocks = 5 # Number of 48-hour blocks to stack (5 blocks = 10 days)

# ---------------------------------------------------------
# Option 1: Stack non-overlapping 48-hour blocks
# ---------------------------------------------------------
stacked_true = []
stacked_pred = []

for i in range(num_blocks):
    sample_idx = i * horizon
    stacked_true.append(y_true[sample_idx, :, node_idx, feature_idx])
    stacked_pred.append(y_pred[sample_idx, :, node_idx, feature_idx])

# Flatten the lists into continuous 1D arrays
continuous_true_blocks = np.concatenate(stacked_true)
continuous_pred_blocks = np.concatenate(stacked_pred)

# ---------------------------------------------------------
# Option 2: 1-Step-Ahead predictions (immediate next hour)
# ---------------------------------------------------------
# Slice the first time step from the number of hours we want to view
total_hours = horizon * num_blocks
step_true = y_true[:total_hours, 0, node_idx, feature_idx]
step_pred = y_pred[:total_hours, 0, node_idx, feature_idx]

# ---------------------------------------------------------
# Plotting both comparisons
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Stacked 48-hour blocks
ax1.plot(continuous_true_blocks, label='True Values', color='black')
ax1.plot(continuous_pred_blocks, label='48h Block Forecasts', color='red', alpha=0.7)
ax1.set_title(f'Stacked 48-Hour Forecasts (Node {node_idx})')
ax1.set_ylabel('Temperature (Normalized)')
ax1.legend()
ax1.grid(True)

# Plot 2: Continuous 1-step-ahead
ax2.plot(step_true, label='True Values', color='black')
ax2.plot(step_pred, label='1-Hour-Ahead Forecast', color='blue', alpha=0.7)
ax2.set_title(f'Continuous 1-Hour-Ahead Forecast (Node {node_idx})')
ax2.set_xlabel('Hours')
ax2.set_ylabel('Temperature (Normalized)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('trajectory_comparison.png')
print("Saved trajectory comparison to 'trajectory_comparison.png'")

df_trajectory = pd.DataFrame({
    'Hour': range(1, len(continuous_true_blocks) + 1),
    'True_Normalized': continuous_true_blocks,
    'Predicted_Normalized': continuous_pred_blocks
})

csv_filename = f'trajectory_node{node_idx}_10days.csv'

# Save it without the pandas index column to keep it clean
df_trajectory.to_csv(csv_filename, index=False)

print(f"Saved readable trajectory to '{csv_filename}'")
