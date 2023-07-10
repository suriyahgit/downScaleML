"""Preprocess ERA-5 data: aggregate to daily data."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import re
import sys
import logging
from logging.config import dictConfig
from joblib import Parallel, delayed

# externals
import xarray as xr

# locals
from downscaleml.core.cli import preprocess_era5_parser
from downscaleml.core.constants import ERA5_VARIABLES
from downscaleml.core.utils import reproject_cdo, search_files, LogConfig
from downscaleml.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = preprocess_era5_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the source directory exists
    if args.source.exists() and args.target:

        # check whether a single variable is specified
        variables = ERA5_VARIABLES
        if args.variable is not None:
            variables = args.variable

        # iterate over the variables to pr- eprocess
        for var in variables:
            # path to files of the current variable
            source = sorted(search_files(
                args.source, '_'.join(['^ERA5', var, '[0-9]{4}.nc$'])))
            ymin, ymax = (re.search('[0-9]{4}', source[0].name)[0],
                          re.search('[0-9]{4}', source[-1].name)[0])
            LogConfig.init_log('Aggregating ERA5 years: {}'.format(
                '-'.join([ymin, ymax])))
            LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
                        ['{}'.format(file) for file in source]))

            # check for dry-run
            if args.dry_run:
                LOGGER.info('Dry run: No output produced.')
                continue

            # check if aggregated file exists
            filename = '_'.join(['ERA5', var, ymin, ymax]) + '.nc'
            filename = args.target.joinpath(var, filename)
            if not filename.parent.exists():
                LOGGER.info('mkdir {}'.format(filename.parent))
                filename.parent.mkdir(parents=True, exist_ok=True)
            if filename.exists() and not args.overwrite:
                LOGGER.info('{} already exists.'.format(filename))
                continue

            # create filenames for resampled files
            dlyavg = [args.target.joinpath(var, f.name.replace('.nc', '_d.nc'))
                      for f in source]

            # aggregate hourly data to daily data: resample in case of missing
            # days
            LOGGER.info('Computing daily averages ...' if var !=
                        'total_precipitation' else 'Computing daily sums ...')
            for src, tmp in zip(source, dlyavg):
                ds = xr.open_dataset(src)

                # compute daily averages/sums
                if var == 'total_precipitation':
                    # convert from m to mm
                    ds = ds.resample(time='D').sum(dim='time') * 1000
                else:
                    ds = ds.resample(time='D').mean(dim='time')

                # save intermediate file for resampling and reprojection to
                # target grid
                ds.to_netcdf(tmp, engine='h5netcdf')

            # reproject and resample to target grid in parallel
            if args.reproject:
                LOGGER.info('Reprojecting and resampling to target grid ...')

                # check whether the target grid file exists
                if not args.grid.exists():
                    LOGGER.info('{} does not exist.'.format(args.grid))
                    sys.exit()

                # create filenames for reprojected files
                target = [args.target.joinpath(var, f.name) for f in source]

                # reproject to target grid
                target = Parallel(n_jobs=-1, verbose=51)(
                        delayed(reproject_cdo)(args.grid, tmp, trg, args.mode,
                                               args.overwrite)
                        for tmp, trg in zip(dlyavg, target))

                # remove temporary daily averages
                for avg in dlyavg:
                    avg.unlink()
            else:
                target = dlyavg

            # aggregate files for different years into a single file using
            # xarray and dask
            LOGGER.info('Aggregating different years into single file ...')
            ds = xr.open_mfdataset(target, concat_dim='time', combine='nested', parallel=True).compute()

            # set NetCDF file compression for each variable
            if args.compress:
                for _, var in ds.data_vars.items():
                    var.encoding['zlib'] = True
                    var.encoding['complevel'] = 5

            # save aggregated netcdf file
            LOGGER.info('Compressing NetCDF: {}'.format(filename))
            ds.to_netcdf(filename, engine='h5netcdf')
            
            # remove single-year files
            if args.purge:
                for trg in target:
                    trg.unlink()
                    
    else:
        LOGGER.info('{} or/and {} does not exist.'.format(str(args.source), str(args.target)))
        
    if args.source.exists() and args.channelprecipi:
        
        # check whether a single variable is specified
        variables = ERA5_VARIABLES
        if args.variable is not None:
            variables = args.variable

        # iterate over the variables to pre-process
        for var in variables:
            # path to files of the current variable
            source = sorted(search_files(
                args.source, '_'.join(['^ERA5', var, '[0-9]{4}.nc$'])))
            ymin, ymax = (re.search('[0-9]{4}', source[0].name)[0],
                          re.search('[0-9]{4}', source[-1].name)[0])
            LogConfig.init_log('Aggregating ERA5 years: {}'.format(
                '-'.join([ymin, ymax])))
            LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
                        ['{}'.format(file) for file in source]))

            # check for dry-run
            if args.dry_run:
                LOGGER.info('Dry run: No output produced.')
                continue

            # check if aggregated file exists
            filename = '_'.join(['ERA5', var, ymin, ymax]) + '.nc'
            filename = args.channelprecipi.joinpath(var, filename)
            filename_resampled = '_'.join(['ERA5', var, ymin, ymax]) +'_resampled' + '.nc'
            filename_resampled = args.channelprecipi.joinpath(var, filename_resampled)
            filename_reprojected = '_'.join(['ERA5', var, ymin, ymax]) +'_reprojected' + '.nc'
            filename_reprojected = args.channelprecipi.joinpath(var, filename_reprojected)
            

            if not filename.parent.exists():
                LOGGER.info('mkdir {}'.format(filename.parent))
                filename.parent.mkdir(parents=True, exist_ok=True)
            if filename.exists() and not args.overwrite:
                LOGGER.info('{} already exists.'.format(filename))
                continue

            # create filenames for resampled files
            dlyavg = [args.channelprecipi.joinpath(var, f.name.replace('.nc', '_d.nc'))
                      for f in source]

            # aggregate files for different years into a single file using
            # xarray and dask
            LOGGER.info('Aggregating different years into single file ...')
            ds = xr.open_mfdataset(source, concat_dim='time', combine='nested', parallel=True)

            # aggregate hourly data to daily data: resample in case of missing
            # days
            LOGGER.info('Computing daily averages ...' if var !=
                        'total_precipitation' else 'Computing daily sums ...')

            ds = ds.shift(time = -6)
            # compute daily averages/sums
            if var == 'total_precipitation':
                # convert from m to mm
                ds = ds.resample(time='D').sum(dim='time') * 1000
            else:
                ds = ds.resample(time='D').mean(dim='time')
            
            last_date = ds.time[-1].values
            ds = ds.drop_sel(time=last_date)

            LOGGER.info('Dataset manipulation done!')

            # save intermediate file for resampling and reprojection to
            # target grid
            ds.load().to_netcdf(filename_resampled, engine='h5netcdf')

            LOGGER.info('Dataset Saved Successfully')

            # reproject and resample to target grid in parallel
            if args.reproject:
                LOGGER.info('Reprojecting and resampling to target grid ...')

                # check whether the target grid file exists
                if not args.grid.exists():
                    LOGGER.info('{} does not exist.'.format(args.grid))
                    sys.exit()

                reprojected_path = reproject_cdo(args.grid, filename_resampled, filename_reprojected, args.mode, args.overwrite)
                LOGGER.info('Reprojection done! and the file saved successfully')
                ds = xr.open_dataset(reprojected_path)
            
            filename_resampled.unlink()

            # set NetCDF file compression for each variable
            if args.compress:
                for _, var in ds.data_vars.items():
                    var.encoding['zlib'] = True
                    var.encoding['complevel'] = 5

            # save aggregated netcdf file
            LOGGER.info('Compressing NetCDF: {}'.format(filename))
            ds.to_netcdf(filename, engine='h5netcdf')
            
            if reprojected_path.exists():
                filename_reprojected.unlink()
    else:
        LOGGER.info('{} or/and {} does not exist.'.format(str(args.source), str(args.channelprecipi)))
        sys.exit()
        
        
