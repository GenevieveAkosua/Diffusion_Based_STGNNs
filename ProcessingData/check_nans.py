import numpy as np

def check_for_nans(file_path="flow.npy"):
    print(f"Loading {file_path}...")
    flow = np.load(file_path)
    
    # Calculate NaN statistics
    total_nans = np.isnan(flow).sum()
    total_elements = flow.size
    nan_percentage = (total_nans / total_elements) * 100
    
    print("-" * 30)
    print(f"Array Shape: {flow.shape}")
    print(f"Total Elements: {total_elements}")
    print(f"Total NaNs: {total_nans}")
    print(f"NaN Percentage: {nan_percentage:.4f}%")
    print("-" * 30)
    
    # If NaNs exist, pinpoint their exact coordinates
    if total_nans > 0:
        print("\nLocation of the first 5 NaNs (Time, Station, Feature):")
        nan_indices = np.argwhere(np.isnan(flow))
        for i in range(min(5, len(nan_indices))):
            t, v, d = nan_indices[i]
            print(f"  -> Time Index: {t}, Station Index: {v}, Feature Index: {d}")
            
        print("\nBreakdown of NaNs by Feature:")
        nans_per_feature = np.isnan(flow).sum(axis=(0, 1))
        for feature_idx, count in enumerate(nans_per_feature):
            print(f"  -> Feature {feature_idx}: {count} NaNs")
    else:
        print("\nData is completely clean! Zero NaNs found.")

if __name__ == "__main__":
    check_for_nans()
