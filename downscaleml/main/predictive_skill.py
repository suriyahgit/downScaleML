# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import sys
import os
import time
import logging
from datetime import timedelta
from logging.config import dictConfig
import numpy as np
import datetime
import pathlib
import pandas as pd
import joblib

# externals
import xarray as xr

from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import r2_score

# locals
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset

from downscaleml.core.config import (ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                     CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                     OVERWRITE, DEM, DEM_FEATURES, STRATIFY,
                                     WET_DAY_THRESHOLD, VALID_SIZE, 
                                     start_year, end_year, CHUNKS)

from downscaleml.core.inputoutput import (NET, ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH)

from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
                                        ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
                                        ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
                                        PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES)

from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
from downscaleml.core.logging import log_conf

print("Import statements performing!")

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize timing
    start_time = time.monotonic()

    # initialize network filename
    state_file = NAMING_Model.state_file(
        NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, dem=DEM,
        dem_features=DEM_FEATURES, doy=DOY)
    
    state_file = '_'.join([state_file, str(start_year)])
    state_file = '_to_'.join([state_file, str(end_year)])

    state_file = MODEL_PATH.joinpath(PREDICTAND, state_file)
    target = TARGET_PATH.joinpath(PREDICTAND)

    # check if output path exists
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    # initialize logging
    log_file = state_file.with_name(state_file.name + "_log.txt")
    
    if log_file.exists():
        log_file.unlink()
    dictConfig(log_conf(log_file))

    # check if target dataset already exists
    target = target.joinpath(state_file.name.replace(state_file.suffix, '.nc'))
    if target.exists() and not OVERWRITE:
        LogConfig.init_log('{} already exists.'.format(target))
        sys.exit()

    LogConfig.init_log('Initializing downscaling for period: {}'.format(
        ' - '.join([str(CALIB_PERIOD[0]), str(CALIB_PERIOD[-1])])))

    # initialize ERA5 predictor dataset
    LogConfig.init_log('Initializing ERA5 predictors.')
    Era5 = ERA5Dataset(ERA5_PATH.joinpath('ERA5'), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    Era5_ds = Era5.merge(chunks=CHUNKS)
    # initialize OBS predictand dataset
    LogConfig.init_log('Initializing observations for predictand: {}'
                       .format(PREDICTAND))

    # read in-situ gridded observations
    Obs_ds = search_files(OBS_PATH.joinpath(PREDICTAND), '.nc$').pop()
    Obs_ds = xr.open_dataset(Obs_ds)

    # whether to use digital elevation model
    if DEM:
        # digital elevation model: Copernicus EU-Dem v1.1
        dem = search_files(DEM_PATH, '^dem_1km.nc$').pop()

        # read elevation and compute slope and aspect
        dem = ERA5Dataset.dem_features(
            dem, {'y': Era5_ds.y, 'x': Era5_ds.x},
            add_coord={'time': Era5_ds.time})

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            dem = dem.drop_vars(['slope', 'aspect']).chunk(Era5_ds.chunks)

        # add dem to set of predictor variables
        dem = dem.chunk(Era5_ds.chunks)
        Era5_ds = xr.merge([Era5_ds, dem])

    # initialize training data
    LogConfig.init_log('Initializing training data.')

    # split calibration period into training and validation period
    if PREDICTAND == 'pr' and STRATIFY:
        # stratify training and validation dataset by number of
        # observed wet days for precipitation
        wet_days = (Obs_ds.sel(time=CALIB_PERIOD).mean(dim=('y', 'x'))
                    >= WET_DAY_THRESHOLD).to_array().values.squeeze()
        train, valid = train_test_split(
            CALIB_PERIOD, stratify=wet_days, test_size=VALID_SIZE)

        # sort chronologically
        train, valid = sorted(train), sorted(valid)
        Era5_train, Obs_train = Era5_ds.sel(time=train), Obs_ds.sel(time=train)
        Era5_valid, Obs_valid = Era5_ds.sel(time=valid), Obs_ds.sel(time=valid)
    else:
        LogConfig.init_log('We are not calculating Stratified Precipitation based on Wet Days here!')

    # training and validation dataset
    Era5_train, Obs_train = Era5_ds.sel(time=CALIB_PERIOD), Obs_ds.sel(time=CALIB_PERIOD)
    Era5_valid, Obs_valid = Era5_ds.sel(time=VALID_PERIOD), Obs_ds.sel(time=VALID_PERIOD)
    
    predictors_train = Era5_train
    predictors_valid = Era5_valid
    predictand_train = Obs_train
    predictand_valid = Obs_valid
    
    # iterate over the grid points
    LogConfig.init_log('Downscaling by Random Forest Starts: iterating each grid cell over time dimension')

    prediction = np.ones(shape=(len(predictors_valid.time), len(predictors_valid.y), len(predictors_valid.x))) * np.nan
    for i, _ in enumerate(predictors_train.x):
        for j, _ in enumerate(predictors_train.y):

            # current grid point: xarray.Dataset, dimensions=(time)
            point_predictors = predictors_train.isel(x=i, y=j)
            point_predictand = predictand_train.isel(x=i, y=j)

            # convert xarray.Dataset to numpy.array: shape=(time, predictors)
            point_predictors = point_predictors.to_array().values.swapaxes(0, 1)
            point_predictand = point_predictand.to_array().values.squeeze()

            # check if the grid point is valid
            if np.isnan(point_predictors).any() or np.isnan(point_predictand).any():
                # move on to next grid point
                continue
            
            # prepare predictors of validation period
            point_validation = predictors_valid.isel(x=i, y=j).to_array().values.swapaxes(0, 1)
            #point_validation = normalize(point_validation)

            predictand_validation = predictand_valid.isel(x=i, y=j).to_array().values.swapaxes(0, 1)

            LogConfig.init_log('Current grid point: ({:d}, {:d})'.format(j, i))    
            # normalize each predictor variable to [0, 1]
            # point_predictors = normalize(point_predictors)

            # instanciate the GLM
            model = TweedieRegressor(power=0 if PREDICTAND in ['tasmax', 'tasmin', 'tasmean'] else 2)
            # power = 0: Normal distribution (tasmax, tasmin)
            # power = 1: Poisson distribution
            # power = (1, 2): Compound Poisson Gamma distribution
            # power = 2: Gamma distribution (pr)
            # power = 3: Inverse Gaussian

            # train model on training data
            model.fit(point_predictors, point_predictand)

            model_file = "{}_{}_{}.joblib".format(str(state_file), j, i)
            # predict validation period
            pred = model.predict(point_validation)
            LogConfig.init_log('Processing grid point: ({:d}, {:d}), score: {:.2f}'.format(j, i, r2_score(predictand_validation, pred)))

            # save model with the index to use it later for any dataset with similar grid
            joblib.dump(model, model_file)
            LogConfig.init_log('Model saved for the current grid point: Saved({:d}, {:d})'.format(j, i))
            
            # store predictions for current grid point
            prediction[:, j, i] = pred
    
    LogConfig.init_log('Model ensemble saved and Indexed')
    
    # store predictions in xarray.Dataset
    predictions = xr.DataArray(data=prediction, dims=['time', 'y', 'x'],
                               coords=dict(time=pd.date_range(predictors_valid.time.values[0],predictors_valid.time.values[-1], freq='D'),
                                           lat=predictand_valid.y, lon=predictand_valid.x))
    predictions = predictions.to_dataset(name=PREDICTAND)

    predictions = predictions.set_index(
        time='time',
        y='lat',
        x='lon'
    )

    # initialize network filename
    predict_file = NAMING_Model.state_file(
        NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, dem=DEM,
        dem_features=DEM_FEATURES, doy=DOY)

    predictions.to_netcdf("{}/{}_{}_to_{}.nc".format(str(target.parent), str(predict_file), str(start_year), str(end_year)))
    
    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE')
            
    
    
    
        

    

