  
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import pathlib
import shutil
from joblib import Parallel, delayed

# externals
import cdsapi
import numpy as np
import subprocess

# CERRA predictor variables on single levels
var = 'total_precipitation'

# path to store downloaded CERRA data
CERRA_PATH = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/CERRA/pr')

# time period
years = [str(y) for y in np.arange(1985, 1987)]
month = [str(m) for m in np.arange(1, 13)]
days = [str(d) for d in np.arange(1, 32)]
time = ["{:02d}:00".format(t) for t in np.arange(0, 24, 3)]

# CERRA download configuration dictionary
CONFIG = {
    'level_type': 'surface',
    'product_type': 'analysis',
    'variable' : 'total_precipitation',
    'month': month,
    'day': days,
    'time': time,
    'format': 'netcdf',
}

# whether to overwrite existing files
OVERWRITE = False

if __name__ == '__main__':

    # initialize client
    c = cdsapi.Client()
    
    # create output directories        
    downloads_output = CERRA_PATH.joinpath('downloads', var)
    processed_output = CERRA_PATH.joinpath('processed', var)
    for dir in [downloads_output, processed_output]:
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)

    # create output files
    files = [downloads_output.joinpath('_'.join(['CERRA', var, year]) + '.nc') for
                 year in years]

    # download configuration: CERRA variable on single level
    product = 'reanalysis-cerra-land'

    # split the download to the different years: CDS API cannot handle
    # requests over the whole time period
    Parallel(n_jobs=min(len(years), os.cpu_count()), verbose=51)(
        delayed(c.retrieve)(
            product, {**CONFIG, **{'year': year}}, file)
        for file, year in zip(files, years) if (not file.exists() or OVERWRITE))


    #These lines below are helpful when you don't have too much of space and also regridding on the go, which saves hard-ships and hard-disks together  
    # post-processing
    for file in files:
        if file.exists():
            # TODO: Add post-processing code here
            # Move file to the processed_output directory
            output_file = processed_output.joinpath(file.stem + '_remapped.nc')
            subprocess.run(['cdo', 'remapbil,/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/interTwin_domain/interTwin_grid', str(file), str(output_file)])
            #os.remove(file)
