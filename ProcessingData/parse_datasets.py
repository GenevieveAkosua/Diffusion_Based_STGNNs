# Author: Genevieve Chikwanha
# Date: 18 July 2026
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


def parse_saws():

	workbook_paths = Path(__file__).parent / 'RawData/saws'
	dataframe_list = []
	# Loop over each workbook(provice) and then over each sheet (feature such as rain, temp, etc)
	for file in workbook_paths.glob('*.xlsx'):
		dataframe_dict = panda.read_excel(file, sheet_name=None, header=None)
		for sheet_name, dataframe in dataframe_dict.items():
			dataframe['source_file'] = file.name # Each file name is a province name
			dataframe['source_sheet'] = sheet_name
			#print(dataframe)
			dataframe_list.append(dataframe)

	combined_dataframe = panda.concat(dataframe_list, ignore_index=True)
	#print(combined_dataframe.head(10))
	#print(combined_dataframe.tail(10))
	#print(combined_dataframe.sample(10))
	#variables = ["Wind Speed", "Wind Direction", "Temperature", "Rain", "Pressure", "Humidity"]
	return combined_dataframe


def parse_era(zipped_file):
	if zipfile.is_zipfile(zipped_file):
		with tempfile.TemporaryDirectory() as tmpdir:
			with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
				zip_ref.extractall(tmpdir)
			nc_files = glob.glob(os.path.join(tmpdir, "*.nc"))
			datasets = [xr.open_dataset(f) for f in nc_files]
			combined_ds = xr.merge(datasets).load()
			for ds in datasets:
				ds.close()
		return combined_ds
	else:
		with xr.open_dataset(zipped_file) as ds:
			return ds.load()



# Map - to NaN, replace , with .
# 2. Regular timestep (from reading)

# Allow handling of missing data by code for ERA5, but use IDW interpolation or something else for SAWS missing data

# Have consistent node ordering with the adj.npy file. Also, save as float32
