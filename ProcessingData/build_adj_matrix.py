# Author: Genevieve Chikwanha
# Date: 24 July 2026
# Purpose: This program builds the adj.npy adjacency matrix required by DiffSTG and USTD,
# using a thresholded Gaussian kernel over pairwise station distances (Li et al. 2018, DCRNN).
# Reads station_metadata.csv (produced by parse_saws.py) and writes adj.npy with node ordering
# matching the "id" column used to build flow_saws.npy.

# TODO: decide between the DCRNN-style auto sigma and the fixed-km air-quality-style version below,
# and confirm which one is actually used for the dissertation.

import numpy as numpy
import pandas as panda
from pathlib import Path


def haversine_km(lat1, lon1, lat2, lon2):
	# Great-circle distance between two lat/lon points, in km
	R = 6371.0
	lat1, lon1, lat2, lon2 = map(numpy.radians, [lat1, lon1, lat2, lon2])
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = numpy.sin(dlat / 2.0) ** 2 + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(dlon / 2.0) ** 2
	c = 2 * numpy.arcsin(numpy.sqrt(a))
	return R * c


def build_distance_matrix(station_metadata):
	# station_metadata must be sorted/indexed by "id" (0..V-1), matching flow.npy's V dimmension
	station_metadata = station_metadata.sort_values("id").reset_index(drop=True)
	n_stations = len(station_metadata)
	latitudes = station_metadata["latitude"].to_numpy()
	longitudes = station_metadata["longitude"].to_numpy()

	dist_matrix = numpy.zeros((n_stations, n_stations), dtype=numpy.float32)
	for i in range(n_stations):
		dist_matrix[i, :] = haversine_km(latitudes[i], longitudes[i], latitudes, longitudes)

	return dist_matrix


def build_adj_dcrnn_style(dist_matrix, epsilon=0.1):
	"""
	DCRNN-style (Li et al. 2018): sigma is the std of all pairwise distances in the dataset,
	so it self-calibrates to however spread out your stations are.
	A_ij = exp(-d_ij^2 / sigma^2) if >= epsilon, else 0
	"""
	sigma = dist_matrix.std()
	adj = numpy.exp(-(dist_matrix ** 2) / (sigma ** 2))
	adj[adj < epsilon] = 0.0
	numpy.fill_diagonal(adj, 0.0)  # DiffSTG's graph_algo.load_graph_data also strips self-loops
	return adj.astype(numpy.float32), sigma


def build_adj_fixed_threshold(dist_matrix, distance_threshold_km=300, sigma_km=100):
	"""
	Air-quality-forecasting style: fixed distance cutoff and fixed sigma (in km), rather than
	deriving sigma from the data. Only keeps edges within distance_threshold_km.
	"""
	adj = numpy.exp(-(dist_matrix ** 2) / (sigma_km ** 2))
	adj[dist_matrix > distance_threshold_km] = 0.0
	numpy.fill_diagonal(adj, 0.0)
	return adj.astype(numpy.float32)


def build_adjacency():
	station_metadata = panda.read_csv("station_metadata.csv")
	dist_matrix = build_distance_matrix(station_metadata)
	adj_dcrnn, sigma = build_adj_dcrnn_style(dist_matrix)
	print(f"DCRNN-style: sigma (std of distances) = {sigma:.2f} km, "
	      f"edges kept = {int((adj_dcrnn > 0).sum())} / {adj_dcrnn.size}")
	numpy.save("adj_dcrnn.npy", adj_dcrnn)

	adj_fixed = build_adj_fixed_threshold(dist_matrix)
	print(f"Fixed-threshold style: edges kept = {int((adj_fixed > 0).sum())} / {adj_fixed.size}")
	numpy.save("adj_fixed.npy", adj_fixed)

	return adj_dcrnn, adj_fixed

adj_dcrnn, adj_fixed = build_adjacency()
print("Finished creating the adjacency matrices)
