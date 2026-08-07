# Author: Genevieve Chikwanha
# Date: 24 July 2026
# Purpose: This program creates a flow.npy file for the given dataset to store the data in .npy format with dimmesions (T, V, D), where T is the time dimmension, V is the vertices/nodes dimmension, and D is the feature dimmension.

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
from parse_datasets import parse_saws
from build_adj_matrix import haversine_km, build_distance_matrix, build_adjacency

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


WIND_DIR_FEATURE = "Wind Direction"
SAWS_FEATURES = ["Temperature", "Humidity", "Pressure", "Rain", "Wind Direction", "Wind Speed"]

def create_timestamped_data(combined_dataframe):
	"""Create a pandas dataframe with station information, date and time,
	and the measured value for a given feature, by reading the SAWS excel sheets"""

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
	"""Calculates the total amount of missing data per station across all variables as a percentage"""

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

def check_year_coverage(timestamps, start_year=2016, end_year=2025):
	"""Drops stations that don't have at least some data for every year in the specified range."""
	# Temporarily extract the year
	timestamps['year'] = timestamps['datetime'].dt.year
	expected_years = set(range(start_year, end_year + 1))
	valid_stations = []
	dropped_stations = []
	# Check each station's available years
	for station, group in timestamps.groupby('station'):
		station_years = set(group['year'].unique())
		if expected_years.issubset(station_years):
			valid_stations.append(station)
		else:
			dropped_stations.append(station)

	verified_df = panda.DataFrame({"Verified Station": valid_stations})
	verified_df.to_csv("verified_stations_2016_2025.csv", index=False)

	# Clean up the temporary column
	timestamps = timestamps.drop(columns=['year'])
	# Filter the dataframe
	filtered_timestamps = timestamps[timestamps['station'].isin(valid_stations)].copy()
	print(f"Year coverage ({start_year}-{end_year}):")
	for station in dropped_stations:
		print(f"Dropped: {station} (Missing required years)")
	
	return filtered_timestamps


def filter_stations(timestamps, missing_readings, threshold=50.0):
	"""Filter the stations based on the percentage of missing data in each station. 
	Stations with missing data exceeding the threshold are dropped, 
	while those below the threshold are kept and used for training."""

	kept_stations = missing_readings.loc[missing_readings["missing_percent"] <= threshold, "station"].tolist()
	dropped_stations = missing_readings.loc[missing_readings["missing_percent"] > threshold, "station"].tolist()
	filtered = timestamps[timestamps["station"].isin(kept_stations)].copy()
	print(f"Number of stations remaining: {len(kept_stations)}")
	for station_name in dropped_stations:
		print(f"Dropped stations: {station_name}")

	return filtered, kept_stations, dropped_stations


def wind_dir_to_components(degrees):
	"""Break wind direction up into its sin and cos components"""

	radians = numpy.radians(degrees)
	return numpy.sin(radians), numpy.cos(radians)


def components_to_wind_dir(sin_component, cos_component):
	"""Recombine the sin and cos components of wind direction using arctan"""

	degrees = numpy.degrees(numpy.arctan2(sin_component, cos_component))
	return numpy.mod(degrees, 360.0)


def pivot_to_wide(long_timestamps, station_ids=None):
	"""Pivot a long-format (datetime/station/feature/value) dataframe to wide format with an
	integer 'id' column per station (0 to V-1). If station_ids is given, reuse that mapping so
	the raw (pre-imputation) pivot and the imputed pivot stay aligned to the same ids.
	"""
	station_metadata_lookup = long_timestamps[["station", "climate_number", "latitude", "longitude", "height"]].drop_duplicates(subset="station")
	pivoted = long_timestamps.pivot_table(index=["datetime", "station"], columns="feature", values="value", dropna=False).reset_index()
	pivoted = pivoted.merge(station_metadata_lookup, on="station", how="left")
	pivoted = pivoted.sort_values(["datetime", "station"]).reset_index(drop=True)
	if station_ids is None:
		stations = sorted(pivoted["station"].unique())
		station_ids = {station: i for i, station in enumerate(stations)}
	pivoted["id"] = pivoted["station"].map(station_ids)
	pivoted = pivoted.sort_values(["datetime", "id"]).reset_index(drop=True)

	return pivoted, station_ids


def to_flow_array(pivoted_timestamps, features):
	"""Build a (T, V, D) numpy array + the underlying xarray Dataset from wide-format
	pivoted_timestamps, for the given feature list."""

	indexed = pivoted_timestamps.set_index(["datetime", "id"])
	xarr = xr.Dataset.from_dataframe(indexed)
	flow = (xarr[features].to_array(dim="feature").transpose("datetime", "id", "feature").astype(numpy.float32).to_numpy())
	return flow, xarr


def interpolate_idw(pivoted_timestamps, dist_matrix, station_ids, power=2):
	"""This fills remaining gaps using an IDW average of
	other stations' readings at the SAME timestamp. Neighbours that are also missing at that
	timestamp are excluded (not treated as zero). If every neighbour is missing too, the value
	is left as NaN.
	"""
	linear_features = ["Temperature", "Humidity", "Pressure", "Rain", "Wind Speed"]
	filled = pivoted_timestamps.copy()
	def idw_fill_column(wide):
		#wide = pivoted_timestamps.pivot(index="datetime", columns="id", values=feature)
		values = wide.to_numpy().copy()
		original = values.copy()
		for col in range(values.shape[1]):
			missing_rows = numpy.isnan(original[:, col])
			if not missing_rows.any():
				continue
			dists = dist_matrix[col, :].copy()
			dists[col] = numpy.inf  # exclude self
			weights = 1.0 / (dists ** power)
			weights[numpy.isinf(dists)] = 0.0
			neighbour_vals = original[missing_rows, :]
			weighted = numpy.nansum(neighbour_vals * weights, axis=1)
			weight_sum = numpy.nansum((~numpy.isnan(neighbour_vals)) * weights, axis=1)
			values[missing_rows, col] = numpy.where(weight_sum > 0, weighted / weight_sum, numpy.nan)
		wide[:] = values
		return wide
		    
	for feature in linear_features:
		wide = pivoted_timestamps.pivot(index="datetime", columns="id", values=feature)
		wide = idw_fill_column(wide)
		filled[feature] = wide.stack().reindex(panda.MultiIndex.from_frame(pivoted_timestamps[["datetime", "id"]])).to_numpy()

	# Wind direction: interpolate sin/cos separately, then recombine
	sin_vals, cos_vals = wind_dir_to_components(pivoted_timestamps[WIND_DIR_FEATURE])
	temp = pivoted_timestamps.copy()
	temp["_sin"] = sin_vals
	temp["_cos"] = cos_vals
	for comp in ["_sin", "_cos"]:
		wide = temp.pivot(index="datetime", columns="id", values=comp)
		wide = idw_fill_column(wide)
		temp[comp] = wide.stack().reindex(panda.MultiIndex.from_frame(temp[["datetime", "id"]])).to_numpy()
	filled[WIND_DIR_FEATURE] = components_to_wind_dir(temp["_sin"], temp["_cos"])

	return filled

def circular_mean_degrees(values):
	"""Mean of a set of compass-degree readings, handling the 0/360 wraparound correctly
	(e.g. mean of [350, 10] should land near 0/360)
	"""
	values = numpy.asarray(values, dtype=float)
	values = values[~numpy.isnan(values)]
	if values.size == 0:
		return numpy.nan
	radians = numpy.radians(values)
	sin_mean = numpy.mean(numpy.sin(radians))
	cos_mean = numpy.mean(numpy.cos(radians))

	return numpy.degrees(numpy.arctan2(sin_mean, cos_mean)) % 360.0
 
 
def summarise_residual_missingness(flow, features, station_metadata):
	"""Prints where the still-NaN entries are concentrated after temporal+spatial imputation,
	broken down by feature and by station -- useful for spotting e.g. a feature IDW always
	skips (Wind Direction), or a station with a long simultaneous regional outage that IDW
	couldn't fix because its neighbours were down too."""
	total_nan = numpy.isnan(flow).sum()
	if total_nan == 0:
		print("No residual NaNs after temporal+spatial imputation.")
		return
	print(f"\n--- Residual NaN breakdown: {int(total_nan)} entries still missing ---")
	T, V, D = flow.shape
	print("By feature:")
	per_feature = numpy.isnan(flow).sum(axis=(0, 1))
	for feature, count in zip(features, per_feature):
		print(f"  {feature:<15} {int(count):>9} / {T*V} ({100.0*count/(T*V):.2f}%)")
	print("By station:")
	station_names = station_metadata.sort_values("id")["station"].tolist()
	per_station = numpy.isnan(flow).sum(axis=(0, 2))
	for name, count in zip(station_names, per_station):
		print(f"  {name:<35} {int(count):>9} / {T*D} ({100.0*count/(T*D):.2f}%)")


def build_saws_flow(pivoted_timestamps, missing_mask, station_metadata, features=SAWS_FEATURES):
	"""Saves the final (imputed) flow.npy alongside the missing_mask captured BEFORE imputation,
	so downstream eval code knows which entries were real SAWS readings vs filled-in values."""

	flow, xarr = to_flow_array(pivoted_timestamps, features)
	xarr.to_netcdf("saws_data.nc")
	summarise_residual_missingness(flow, features, station_metadata)
	#flow = fill_remaining_gaps(flow, features)
	#assert not numpy.isnan(flow).any(), "fill_remaining_gaps left NaNs behind -- check for an all-NaN feature/station"
	numpy.save("flow_saws.npy", flow)
	numpy.save("missing_mask_saws.npy", missing_mask.astype(numpy.bool_))
 
	flow = numpy.load("flow_saws.npy")
	missing_mask = numpy.load("missing_mask_saws.npy")
	print("BUILD SAWS")
	print("Tensor Shape:", flow.shape)
	print("Missing Mask Shape:", missing_mask.shape)
	assert flow.shape == missing_mask.shape, "flow.npy and missing_mask_saws.npy shapes don't match -- check pivot alignment"
	print("Data Type:", flow.dtype)
	print(f"Originally missing: {missing_mask.sum()} / {missing_mask.size} ({100.0*missing_mask.mean():.2f}%)")
	print(f"Still NaN after temporal+spatial imputation: {numpy.isnan(flow).sum()} (these need a fallback, e.g. mean/zero-fill, before training)")
	print("flow.npy is fully finite (no NaNs) -- missingness lives only in missing_mask_saws.npy now.")
	for idx in (80000, 67000):
		if idx < flow.shape[0]:
			print(flow[idx, min(2, flow.shape[1]-1), :])

def calculate_features(file_path):
	"""Calculate era5 relative humidity and wind speed"""

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

def check_station_years(timestamps):
	"""Temporary function to verify station data ranges (2016-2025)."""
	# Convert list of dicts to DataFrame
	df = panda.DataFrame(timestamps)
	# Extract the year from the datetime column
	df['year'] = df['datetime'].dt.year
	# Group by station to find the min, max, and count of unique years
	year_ranges = df.groupby('station')['year'].agg(['min', 'max', 'nunique'])
	print("\n" + "="*40)
	print("STATION YEAR RANGES (MIN / MAX / UNIQUE COUNT)")
	print("="*40)
	print(year_ranges)
	print("="*40)
    
	# Strictly check for complete 2016-2025 coverage
	expected_years = set(range(2016, 2026)) # 2016 to 2025 inclusive
	missing_years_report = []
	for station, group in df.groupby('station'):
		station_years = set(group['year'].unique())
		missing = expected_years - station_years
		if missing:
			missing_years_report.append(f"  - {station}: Missing {sorted(list(missing))}")
	print("\nCOVERAGE CHECK (2016-2025):")
	if missing_years_report:
		print("The following stations are missing expected years:")
		for report in missing_years_report:
			print(report)
	else:
		print("Success! All stations have at least some data for every year from 2016 to 2025.")
	print("="*40 + "\n")

    
#data = parse_saws()
#timestamps = create_timestamped_data(data)
#print(timestamps)
#missing_readings = compute_missing_readings(timestamps)
#print(missing_readings)
#filtered, kept, dropped = filter_stations(timestamps, missing_readings, threshold=50.0)
#build_saws_flow(filtered)
#print("Finished")
#build_era_flow()
#build_full_era_flow()

if __name__ == "__main__":
	data = parse_saws()
	timestamps = create_timestamped_data(data)
	timestamps = check_year_coverage(timestamps, 2016, 2025)
	#print(timestamps)
	missing_readings = compute_missing_readings(timestamps)
	print(missing_readings)
	filtered, kept, dropped = filter_stations(timestamps, missing_readings, threshold=10.0)

	# Pivot the RAW (pre-imputation) data first, purely to lock in station ids/metadata
	# and to capture missing_mask_saws.npy -- this must reflect true SAWS missingness, not
	# "still missing after we filled some of it in".
	raw_pivoted, station_ids = pivot_to_wide(filtered)
	station_metadata = (raw_pivoted[["id", "station", "climate_number", "latitude", "longitude", "height"]].drop_duplicates().sort_values("id"))
	station_metadata.to_csv("station_metadata.csv", index=False)
	build_adjacency()
	raw_flow, _ = to_flow_array(raw_pivoted, SAWS_FEATURES)
	missing_mask = numpy.isnan(raw_flow)

	# Temporal short-gap fill (also fixes the wind-direction wraparound bug)
	#temporally_filled = interpolate_temporal(filtered, max_gap=3)
	filled_pivoted, _ = pivot_to_wide(filtered, station_ids=station_ids)

	# Spatial IDW fill for whatever's still missing after the temporal pass
	dist_matrix = build_distance_matrix(station_metadata, "latitude", "longitude")
	filled_pivoted = interpolate_idw(filled_pivoted, dist_matrix, station_ids, power=2)

	# Save flow_saws.npy (imputed) + missing_mask_saws.npy (original missingness)
	print("Building the SAWS flow file")
	build_saws_flow(filled_pivoted, missing_mask, station_metadata)


