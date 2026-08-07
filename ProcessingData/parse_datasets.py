# Author: Genevieve Chikwanha
# Date: 18 July 2026
# Purpose: This program reads in a SAWS csv file into a pandas dataframe

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
	for file in workbook_paths.glob('*.xls*'):
		dataframe_dict = panda.read_excel(file, sheet_name=None, header=None)
		for sheet_name, dataframe in dataframe_dict.items():
			dataframe['source_file'] = file.name # Each file name is a province name
			dataframe['source_sheet'] = sheet_name
			dataframe_list.append(dataframe)

	combined_dataframe = panda.concat(dataframe_list, ignore_index=True)
	return combined_dataframe

if __name__ == "__main__":
	parse_saws()
	print("Parse successful")
