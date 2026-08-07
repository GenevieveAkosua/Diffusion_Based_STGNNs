# Author: Genevieve Chikwanha
# Date: 24 July 2026
# Purpose: This program builds the adj.npy adjacency matrix required by DiffSTG and USTD,
# using a thresholded Gaussian kernel over pairwise station distances (Li et al. 2018, DCRNN).
# Reads station_metadata.csv (produced by parse_saws.py) and writes adj.npy with node ordering
# matching the "id" column used to build flow_saws.npy.

import numpy as numpy
import pandas as panda
from pathlib import Path


def haversine_km(lat1, lon1, lat2, lon2):
	"""Great-circle distance between two lat/lon points, in km"""

	R = 6371.0
	lat1, lon1, lat2, lon2 = map(numpy.radians, [lat1, lon1, lat2, lon2])
	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = numpy.sin(dlat / 2.0) ** 2 + numpy.cos(lat1) * numpy.cos(lat2) * numpy.sin(dlon / 2.0) ** 2
	c = 2 * numpy.arcsin(numpy.sqrt(a))
	return R * c


def build_distance_matrix(station_metadata, latitude, longitude):
	"""Station_metadata must be sorted/indexed by "id" (0 to V-1), matching flow.npy's V dimmension"""

	station_metadata = station_metadata.sort_values("id").reset_index(drop=True)
	n_stations = len(station_metadata)
	latitudes = station_metadata[latitude].to_numpy()
	longitudes = station_metadata[longitude].to_numpy()
	dist_matrix = numpy.zeros((n_stations, n_stations), dtype=numpy.float32)
	for i in range(n_stations):
		dist_matrix[i, :] = haversine_km(latitudes[i], longitudes[i], latitudes, longitudes)

	return dist_matrix

def build_coordinate_matrix(station_metadata, latitude, longitude):
	"""Builds a 2D array of shape [Stations, 2] where each row is the geographic location 
	[latitude, longitude] of a station for CLCRN"""

	station_metadata = station_metadata.sort_values("id").reset_index(drop=True)
	n_stations = len(station_metadata)
	latitudes = station_metadata[latitude].to_numpy()
	longitudes = station_metadata[longitude].to_numpy()
	coord_matrix = numpy.stack([latitudes, longitudes], axis=1).astype(numpy.float32)
	numpy.save("coordinates.npy", coord_matrix)


def build_spatial_corr_matrix(dist_matrix, epsilon=0.1):
	"""Builds a 2D spatial correlation matrix for DiffSTG based on DCRNN's implementation.
	"""
	
	sigma = float(dist_matrix.std())
	adj = numpy.exp(-(dist_matrix ** 2) / (sigma ** 2))
	adj[adj < epsilon] = 0.0
	numpy.fill_diagonal(adj, 0.0)
	adj_matrix = adj.astype(numpy.float32)
	numpy.save("adj.npy", adj_matrix)


def build_adjacency():
	station_metadata = panda.read_csv("station_metadata.csv")
	dist_matrix = build_distance_matrix(station_metadata, "latitude", "longitude")
	build_spatial_corr_matrix(dist_matrix)
	build_coordinate_matrix(station_metadata, "latitude", "longitude")

if __name__ == "__main__":
	build_adjacency()
	print("Finished creating the adjacency matrices")
