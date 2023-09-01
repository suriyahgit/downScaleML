"""Download CERRA data from the Copernicus Climate Data Store."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import pathlib
from joblib import Parallel, delayed

# externals
import cdsapi
import numpy as np

# ERA5 predictor variables on pressure levels
var = 'specific_humidity'

# path to store downloaded ERA5 data
ERA5_PATH = pathlib.Path('/home/sdhinakaran/temp_location/')

# time period
years = [str(y) for y in np.arange(2017, 2021)]
month = [str(m) for m in np.arange(1, 13)]
days = [str(d) for d in np.arange(1, 32)]
time = ["{:02d}:00".format(t) for t in np.arange(0, 24)]

area = [52, 2, 40, 20]

# ERA5 download configuration dictionary
CONFIG = {
    'product_type': 'reanalysis',
    'format': 'netcdf',
    'month': month,
    'day': days,
    'time': time,
    'format': 'netcdf',
    'pressure_level': [
            '250', '500', '700',
            '850', '1000',
        ],
    'variable': 'specific_humidity',
    'area': area,
}

# whether to overwrite existing files
OVERWRITE = False

# whether to skip variables on single levels
SKIP_SINGLE_LEVELS = True

if __name__ == '__main__':

    # initialize client
    c = cdsapi.Client()
    
    # create output directory        
    output = ERA5_PATH
    if not output.exists():
        output.mkdir(parents=True, exist_ok=True)

    # create output files
    files = [output.joinpath('_'.join(['ERA5', var, year]) + '.nc') for
                 year in years]

    # download configuration: ERA5 variable on single level
    product = 'reanalysis-era5-pressure-levels'

    # split the download to the different years: CDS API cannot handle
    # requests over the whole time period
    Parallel(n_jobs=min(len(years), os.cpu_count()), verbose=51)(
        delayed(c.retrieve)(
            product, {**CONFIG, **{'year': year}}, file)
        for file, year in zip(files, years) if (not file.exists() or OVERWRITE))
