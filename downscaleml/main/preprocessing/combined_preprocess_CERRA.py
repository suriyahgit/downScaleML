"""Preprocess CERRA data: aggregate to daily data."""

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
from downscaleml.core.cli import preprocess_cerra_parser
from downscaleml.core.constants import CERRA_VARIABLES
from downscaleml.core.utils import reproject_cdo, search_files, LogConfig
from downscaleml.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = preprocess_cerra_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)
        
    if args.source.exists() and args.target:
        
        # check whether a single variable is specified
        variables = CERRA_VARIABLES
        if args.variable is not None:
            variables = args.variable

        # iterate over the variables to pre-process
        for var in variables:
            # path to files of the current variable
            source = sorted(search_files(
                args.source, '_'.join(['^CERRA', var, '[0-9]{4}.nc$'])))
            ymin, ymax = (re.search('[0-9]{4}', source[0].name)[0],
                          re.search('[0-9]{4}', source[-1].name)[0])
            LogConfig.init_log('Aggregating CERRA years: {}'.format(
                '-'.join([ymin, ymax])))
            LOGGER.info(('\n ' + (len(__name__) + 1) * ' ').join(
                        ['{}'.format(file) for file in source]))

            # check for dry-run
            if args.dry_run:
                LOGGER.info('Dry run: No output produced.')
                continue

            # check if aggregated file exists
            filename = '_'.join(['CERRA', var, ymin, ymax]) + '.nc'
            filename = args.target.joinpath(var, filename)
            filename_resampled = '_'.join(['CERRA', var, ymin, ymax]) +'_resampled' + '.nc'
            filename_resampled = args.target.joinpath(var, filename_resampled)
            filename_reprojected = '_'.join(['CERRA', var, ymin, ymax]) +'_reprojected' + '.nc'
            filename_reprojected = args.target.joinpath(var, filename_reprojected)
            
            if var == "2m_temperature":
                # check if aggregated file exists
                filename_tasmin = '_'.join(['CERRA_tasmin', var, ymin, ymax]) + '.nc'
                filename_tasmin = args.target.joinpath(var, filename_tasmin)
                filename_resampled_tasmin = '_'.join(['CERRA_tasmin', var, ymin, ymax]) +'_resampled' + '.nc'
                filename_resampled_tasmin = args.target.joinpath(var, filename_resampled_tasmin)
                filename_reprojected_tasmin = '_'.join(['CERRA_tasmin', var, ymin, ymax]) +'_reprojected' + '.nc'
                filename_reprojected_tasmin = args.target.joinpath(var, filename_reprojected_tasmin)# check if aggregated file exists
                
                filename_tasmax = '_'.join(['CERRA_tasmax', var, ymin, ymax]) + '.nc'
                filename_tasmax = args.target.joinpath(var, filename_tasmax)
                filename_resampled_tasmax = '_'.join(['CERRA_tasmax', var, ymin, ymax]) +'_resampled' + '.nc'
                filename_resampled_tasmax = args.target.joinpath(var, filename_resampled_tasmax)
                filename_reprojected_tasmax = '_'.join(['CERRA_tasmax', var, ymin, ymax]) +'_reprojected' + '.nc'
                filename_reprojected_tasmax = args.target.joinpath(var, filename_reprojected_tasmax)
            

            if not filename.parent.exists():
                LOGGER.info('mkdir {}'.format(filename.parent))
                filename.parent.mkdir(parents=True, exist_ok=True)
            if filename.exists() and not args.overwrite:
                LOGGER.info('{} already exists.'.format(filename))
                continue

            # create filenames for resampled files
            # dlyavg = [args.target.joinpath(var, f.name.replace('.nc', '_d.nc'))
                      # for f in source]

            # aggregate files for different years into a single file using
            # xarray and dask
            LOGGER.info('Aggregating different years into single file ...')
            ds = xr.open_mfdataset(source, concat_dim='time', combine='nested', parallel=True)
 
            # aggregate hourly data to daily data: resample in case of missing
            # days
            LOGGER.info('Computing daily averages, min and max for temperature ...' if var !=
                        'total_precipitation' else 'Time shifting the Precipitation as accumulation is stored at the following day 6AM ...')

            # compute daily averages/sums
            if var == 'total_precipitation':
                #CERRA Total Precipitation is in kg/m-2, it's equal to mm
                ds = ds.resample(time='1D').first()
                ds = ds.shift(time = -1)
                last_date = ds.time[-1].values
                ds = ds.drop_sel(time=last_date)

                LOGGER.info('Total Precipitation Dataset manipulation done!')

                if not filename_resampled.exists():
                    ds.to_netcdf(filename_resampled, engine='h5netcdf')
                
            if var == "2m_temperature":
                if not (filename_resampled.exists() or filename_resampled_tasmax.exists() or filename_resampled_tasmin.exists()):
                    ds = ds.resample(time='D').mean(dim='time')
                    ds.to_netcdf(filename_resampled, engine='h5netcdf')
                    del ds
                    LOGGER.info('tasmean Dataset manipulation done!')
                    ds_max = ds_max.resample(time='D').max(dim='time')
                    ds_max.to_netcdf(filename_resampled_tasmax, engine='h5netcdf')
                    del ds_max
                    LOGGER.info('tasmax Dataset manipulation done!')
                    ds_min = ds_min.resample(time='D').min(dim='time')
                    ds_min.to_netcdf(filename_resampled_tasmin, engine='h5netcdf')
                    del ds_min
                    LOGGER.info('tasmin Dataset manipulation done!')
                    LOGGER.info('Check if it reached here!')

                LOGGER.info('Dataset Saved Successfully')
                ds = xr.open_dataset(str(filename_resampled))
                ds_min = xr.open_dataset(str(filename_resampled_tasmin))
                ds_max = xr.open_dataset(str(filename_resampled_tasmax))

                LOGGER.info('Dataset loaded Successfully, this is done to offload memory in RAM while resampling!')

            # reproject and resample to target grid in parallel
            if args.reproject:
                LOGGER.info('Reprojecting and resampling to target grid ...')

                # check whether the target grid file exists
                if not args.grid.exists():
                    LOGGER.info('{} does not exist.'.format(args.grid))
                    sys.exit()
                
                reprojected_path = reproject_cdo(args.grid, filename_resampled, filename_reprojected, args.mode, args.overwrite)
                
                if var == "2m_temperature":
                    reprojected_path_tasmax = reproject_cdo(args.grid, filename_resampled_tasmax, filename_reprojected_tasmax, args.mode, args.overwrite)
                    ds_max = xr.open_dataset(reprojected_path_tasmax)
                    reprojected_path_tasmin = reproject_cdo(args.grid, filename_resampled_tasmin, filename_reprojected_tasmin, args.mode, args.overwrite)
                    ds_min = xr.open_dataset(reprojected_path_tasmin)
                
                LOGGER.info('Reprojection done! and the file saved successfully')
                ds = xr.open_dataset(reprojected_path)
            
                filename_resampled.unlink()
                if var == "2m_temperature":
                    filename_resampled_tasmax.unlink()
                    filename_resampled_tasmin.unlink()

            # set NetCDF file compression for each variable
            if args.compress:
                for _, varia in ds.data_vars.items():
                    varia.encoding['zlib'] = True
                    varia.encoding['complevel'] = 5
                    
                if var == "2m_temperature":
                    for _, variab in ds_max.data_vars.items():
                        variab.encoding['zlib'] = True
                        variab.encoding['complevel'] = 5
                    
                    for _, vari in ds_min.data_vars.items():
                        vari.encoding['zlib'] = True
                        vari.encoding['complevel'] = 5

                    if filename_resampled_tasmax.exists():
                        filename_resampled_tasmax.unlink()

                    if filename_resampled_tasmin.exists():
                        filename_resampled_tasmax.unlink()
                
                if filename_reprojected.exists():
                    filename_reprojected.unlink()
                    
                if var == "2m_temperature":
                    if filename_reprojected_tasmax.exists():
                        filename_reprojected_tasmax.unlink()
                    if filename_reprojected_tasmin.exists():
                        filename_reprojected_tasmin.unlink()

            # save aggregated netcdf file
            LOGGER.info('Compressing NetCDF: {}'.format(filename))
            ds.load().to_netcdf(filename, engine='h5netcdf')
            if var == "2m_temperature":
                ds_max.to_netcdf(filename_tasmin, engine='h5netcdf')
                ds_min.to_netcdf(filename_tasmax, engine='h5netcdf')

            LOGGER.info('If you reached here, Have a coffee break!')
            
    else:
        LOGGER.info('{} or/and {} does not exist.'.format(str(args.source), str(args.target)))
        sys.exit()
        
        
