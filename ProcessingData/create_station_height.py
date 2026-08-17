# -*- coding: utf-8 -*-
# Builds dataset/saws/station_heights.npy from station_metadata.csv
# Node ordering in the CSV's `id` column must match adj.npy / flow.npy ordering.

import numpy as np
import pandas as pd

CSV_PATH = 'station_metadata.csv'
OUT_PATH = 'station_heights.npy'

df = pd.read_csv(CSV_PATH)

required_cols = {'id', 'station', 'height'}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f'CSV is missing expected column(s): {missing_cols}')

# --- validate id column forms a complete, contiguous 0..N-1 sequence ---
ids = df['id'].to_numpy()
num_nodes = len(df)
expected_ids = np.arange(num_nodes)

if set(ids) != set(expected_ids.tolist()):
    raise ValueError(
        f"id column is not a contiguous 0..{num_nodes - 1} sequence. "
        f"Got ids: {sorted(ids.tolist())}"
    )

# --- sort by id to guarantee output ordering matches adj/flow node order ---
# (defensive: file already appears sorted, but don't assume row order == id order)
df_sorted = df.sort_values('id').reset_index(drop=True)

if not np.array_equal(df_sorted['id'].to_numpy(), expected_ids):
    raise ValueError('Failed to align id column to 0..N-1 after sorting — check for duplicate ids.')

# --- check for missing height values ---
if df_sorted['height'].isna().any():
    bad_stations = df_sorted.loc[df_sorted['height'].isna(), 'station'].tolist()
    raise ValueError(f'Missing height value(s) for station(s): {bad_stations}')

station_heights = df_sorted['height'].to_numpy().astype(np.float32)

np.save(OUT_PATH, station_heights)

print(f'Saved {OUT_PATH}: shape={station_heights.shape}, dtype={station_heights.dtype}')
print(f'Height range: {station_heights.min():.1f}m to {station_heights.max():.1f}m')
print('\nNode order check (id -> station -> height):')
for i, row in df_sorted.iterrows():
    print(f'  {i:2d}  {row["station"]:<35s} {row["height"]:.1f}m')
