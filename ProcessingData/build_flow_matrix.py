# Author: Genevieve Chikwanha
# Date: 24 July 2026
# Purpose: This program creates a flow.npy file for the given dataset to store the data in .npy format with dimmesions (T, V, D), where T is the time dimmension, V is the vertices/nodes dimmension, and D is the feature dimmension.

# TODO: BE SURE TO DEAL WITH NaN DATA AND ALSO IMPUTE DATA!!!

import numpy as numpy
import pandas as panda
import xarray as xr
from pathlib import Path
import zipfile
import tempfile
import glob
import os
import re
from calendar import monthrange
from datetime import datetime
from parse_datasets import parse_saws, parse_era

# Read in SAWS data
# SAWS data is in the format:
#
# |-- Province (the entire excel spreadsheet)
# 	  |-- Features (the sheets)
#     	  |--Stations
#		     |--Day (cols) and Time (rows)

# The goal is to get the data in the format:
# |-- Day and Time
#     |-- Stations
#		 |-- Features


def create_timestamped_data(combined_dataframe):
	date_rows = combined_dataframe[combined_dataframe[0].str.startswith("HOURLY DATA", na=False)].index.tolist()

	timestamps = []

	# Get the timestamps (date, time, and station metadata)
	for date_index in date_rows:
		
		# Get the month and year
		text = combined_dataframe.iloc[date_index, 0]
		match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", str(text))
		month_name = match.group(1)
		year = int(match.group(2))
		month = datetime.strptime(month_name, "%B").month
		n_days = monthrange(year, month)[1]

		# Extract station data
		text = combined_dataframe.iloc[date_index + 1, 0]
		station = re.search(r"^(.*?)\s+-\s+Climate", str(text))
		station = station.group(1)
		climate = re.search(r"Climate Number:\s*([0-9 ]+)", str(text))
		climate_number = climate.group(1).strip()
		lat = re.search(r"Lat:([-\d.]+)", str(text))
		latitude = float(lat.group(1))
		lon = re.search(r"Lon:([-\d.]+)", str(text))
		longitude = float(lon.group(1))
		height = re.search(r"Height:([-\d.]+)", str(text))
		height = float(height.group(1))

		# Extract weather variable/feature
		feature = combined_dataframe.loc[date_index, "source_sheet"]
		# Loop through the days (rows)
		curr_row = date_index + 3
		while curr_row < len(combined_dataframe):
			first_cell = str(combined_dataframe.iloc[curr_row, 0]).strip()
			cell = first_cell.lower()
			if cell == "avg" or cell == "max" or cell == "tot" or not cell.isdigit():
				break
			day = int(first_cell)

			# Loop through the columns (hours) for each day to create 
			for hour in range(24):
				value = str(combined_dataframe.iloc[curr_row, hour + 1]).strip()
				# If data is missing, set to NaN, if blank then set to 0, else replace "," with "."
				if value in ["-", "***", "---", "="]:
					value = numpy.nan
				elif value == "":
					value = 0.0
				else:
					value = float(value.replace(",", "."))
				timestamp = panda.Timestamp(year=year, month=month, day=day, hour=hour)
				timestamps.append({"datetime": timestamp, "station": station, "climate_number": climate_number, "latitude": latitude, "longitude": longitude, "height": height, "feature": feature, "value": value})

			curr_row += 1

	return panda.DataFrame(timestamps)


def compute_missing_readings(timestamps):
	grouped = timestamps.groupby("station")
	summary_rows = []
	# Loop through the stations to calculate the amount of missing data
	for station_name, station_group in grouped:
		total_readings = len(station_group)
		missing_readings = station_group["value"].isna().sum()
		missing_percent = 100.0 * missing_readings / total_readings
		summary_rows.append({"station": station_name, "total_readings": total_readings, "missing_reading": missing_readings, "missing_percent": missing_percent})
	missing_readings = panda.DataFrame(summary_rows).sort_values("missing_percent", ascending=False)
	return missing_readings


def filter_stations(timestamps, missing_readings, threshold=50.0):
	kept_stations = missing_readings.loc[missing_readings["missing_percent"] <= threshold, "station"].tolist()
	dropped_stations = missing_readings.loc[missing_readings["missing_percent"] > threshold, "station"].tolist()
	filtered = timestamps[timestamps["station"].isin(kept_stations)].copy()
	for station_name in dropped_stations:
		print(f"Dropped stations: {station_name}")
	return filtered, kept_stations, dropped_stations


def build_saws_flow(filtered_timestamps):
	# Pivots rows from single measurement to each row being a station and feats being cols
	pivoted_timestamps = filtered_timestamps.pivot_table(index=["datetime", "station", "climate_number", "latitude", "longitude", "height"], columns="feature", values="value").reset_index()
	pivoted_timestamps = pivoted_timestamps.sort_values(["datetime", "station"]).reset_index(drop=True)
	stations = sorted(pivoted_timestamps["station"].unique())
	station_ids = {}
	for i, station in enumerate(stations):
		station_ids[station] = i
	pivoted_timestamps["id"] = pivoted_timestamps["station"].map(station_ids)
	station_metadata = (pivoted_timestamps[["id", "station", "climate_number", "latitude", "longitude", "height"]].drop_duplicates().sort_values("id"))
	station_metadata.to_csv("station_metadata.csv", index=False)
	pivoted_timestamps = pivoted_timestamps.sort_values(["datetime", "id"])
	indexed = pivoted_timestamps.set_index(["datetime", "id"])
	xarr = xr.Dataset.from_dataframe(indexed)
	xarr.to_netcdf("saws_data.nc")
	features = ["Temperature", "Humidity", "Pressure", "Rain", "Wind Direction", "Wind Speed"]
	flow = (xarr[features].to_array(dim="feature").transpose("datetime", "id", "feature").astype(numpy.float32).to_numpy())
	numpy.save("flow_saws.npy", flow)
	flow = numpy.load("flow_saws.npy")
	print("BUILD SAWS")
	print("Tensor Shape:", flow.shape)
	print("Data Type:", flow.dtype)
	print(flow[80000, 2, :])
	print(flow[67000, 1, :])


def calculate_features(file_path):
	data = parse_era(file_path)
	data.to_netcdf(file_path)
	# Converting temp and dewpoint to celcius (from kelvin), rain to mm (from m), and pressure to millibars (mb)
	data['t2m_c'] = data['t2m'] - 273.15
	data['d2m_c'] = data['d2m'] - 273.15
	data['sp_mb'] = data['sp'] / 100.00
	data['tp_mm'] = data['tp'] * 1000.00
	# Calculate wind speed and relative humidity
	data['ws10'] = numpy.sqrt(data['u10']**2 + data['v10']**2)
	numerator = numpy.exp((17.625 * data['d2m_c']) / (243.04 + data['d2m_c']))
	denomenator = numpy.exp((17.625 * data['t2m_c']) / (243.04 + data['t2m_c']))
	data['rh'] = 100.00 * (numerator / denomenator)
	return data

def build_era_flow():
	# Unzips all the era files and builds the flow graph
	all_files = glob.glob("RawData/era/*.nc")
	flow_matrices = []
	for file_path in sorted(all_files):
		data = calculate_features(file_path)
		saws_metadata = panda.read_csv("station_metadata.csv")
		station_coords = list(zip(saws_metadata['latitude'], saws_metadata['longitude']))
		print(station_coords)
		features = ['t2m_c', 'tp_mm', 'sp_mb', 'ws10', 'rh', 'u10', 'v10', 'd2m_c']
		id_arrays = []
		# Find the coordinates correspodning with those in the saws data
		for lat, lon in station_coords:
			station_data = data[features].sel(latitude=lat, longitude=lon, method='nearest')
			station_data_arr = station_data.to_array().values.T
			id_arrays.append(station_data_arr)
		flow_matrix = numpy.stack(id_arrays, axis=1)
		flow_matrices.append(flow_matrix)
		data.close()
		print("BUILD ERA")
		print(f"{flow_matrix.shape}")
	concat_flow_matrices = numpy.concat(flow_matrices, axis=0)
	numpy.save("flow_era.npy", concat_flow_matrices)
	flow = numpy.load("flow_era.npy")
	print("Tensor Shape:", flow.shape)
	print("Data Type:", flow.dtype)
	print(flow[0, 2, :])
	print(flow[700, 1, :])


def build_full_era_flow():
	all_files = glob.glob("RawData/era/*.nc")
	flow_matrices = []
	for file_path in sorted(all_files):
		data = calculate_features(file_path)
		data = data.isel(latitude=slice(None, None, 2), longitude=slice(None, None, 2))
		data_stacked = data.stack(id=("latitude", "longitude"))
		latitudes = data_stacked.latitude.values
		longitudes = data_stacked.longitude.values
		era_metadata = panda.DataFrame({"id": range(len(latitudes)), "latitude": latitudes, "longitude": longitudes})
		print(era_metadata)
		era_metadata.to_csv("era_grid_metadata.csv", index=False)
		features = ['t2m_c', 'tp_mm', 'sp_mb', 'ws10', 'rh', 'u10', 'v10', 'd2m_c']
		flow_matrix = data_stacked[features].to_array().transpose('valid_time', 'id', 'variable').astype(numpy.float32).values
		data.close()
		flow_matrices.append(flow_matrix)
	concat_flow_matrices = numpy.concat(flow_matrices, axis=0)
	numpy.save("flow_era_full.npy", concat_flow_matrices)
	print("BUILD FULL ERA")
	print("Full ERA Grid Tensor Shape:", concat_flow_matrices.shape)
    

data = parse_saws()
timestamps = create_timestamped_data(data)
print(timestamps)
missing_readings = compute_missing_readings(timestamps)
print(missing_readings)
filtered, kept, dropped = filter_stations(timestamps, missing_readings, threshold=50.0)
build_saws_flow(filtered)
print("Finished")
build_era_flow()
build_full_era_flow()

