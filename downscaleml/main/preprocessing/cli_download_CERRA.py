  
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import os
import pathlib
import shutil
from joblib import Parallel, delayed
import logging
from logging.config import dictConfig
import sys

# externals
import cdsapi
import numpy as np
import subprocess

from downscaleml.core.cli import batch_download_parser
from downscaleml.core.utils import LogConfig
from downscaleml.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)

if __name__ == '__main__':
    
    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = batch_download_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)
        
    if args.variable is not None:
        var = args.variable
        var = var[0]
    else:
        LOGGER.info('{} does not exist.'.format(args.variable))
        sys.exit()
    
    if not args.startyear:
        LOGGER.info('{} does not exist.'.format(args.startyear))
        sys.exit()
    
    if not args.endyear:
        LOGGER.info('{} does not exist.'.format(args.endyear))
        sys.exit()
    
    if isinstance(args.target, pathlib.Path):
        LOGGER.info("{} is a pathlib path object.".format(args.target))
        CERRA_PATH = args.target
    else:
        LOGGER.info("{} is not a pathlib path object.".format(args.target))
        sys.exit()

    # time period
    years = [str(y) for y in np.arange(args.startyear, (args.endyear+1))]
    month = [str(m) for m in np.arange(1, 13)]
    days = [str(d) for d in np.arange(1, 32)]
    time = ["{:02d}:00".format(t) for t in np.arange(0, 24, 3)]

    if var == "total_precipitation":
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
        product = 'reanalysis-cerra-land'
    
    if var == "2m_temperature":
        # CERRA download configuration dictionary
        CONFIG = {
            'level_type': 'surface_or_athmosphere',
            'product_type': 'analysis',
            'variable' : '2m_temperature',
            'data_type': 'reanalysis',
            'month': month,
            'day': days,
            'time': time,
            'format': 'netcdf',
        }
        product = 'reanalysis-cerra-single-levels'

    # whether to overwrite existing files
    OVERWRITE = False

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

    # split the download to the different years: CDS API cannot handle
    # requests over the whole time period
    Parallel(n_jobs=min(len(years), os.cpu_count()), verbose=51)(
        delayed(c.retrieve)(
            product, {**CONFIG, **{'year': year}}, file)
        for file, year in zip(files, years) if (not file.exists() or OVERWRITE))


    # These lines below are helpful when you don't have too much of space and also regridding on the go, which saves hard-ships and hard-disks together  
    # post-processing
    if args.reproject and args.grid:
        for file in files:
            if file.exists():
                # TODO: Add post-processing code here
                # Move file to the processed_output directory
                output_file = processed_output.joinpath(file.stem + '_remapped.nc')
                subprocess.run(['cdo', 'remapbil,{}'.format(args.grid), str(file), str(output_file)])
                
                # If you want to remove the original files
                if args.purge:
                    os.remove(file)
