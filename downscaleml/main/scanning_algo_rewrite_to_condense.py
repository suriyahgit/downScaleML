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

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# locals
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset

from downscaleml.main.config import (NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                     CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                     OVERWRITE, DEM, DEM_FEATURES, STRATIFY,
                                     WET_DAY_THRESHOLD, VALID_SIZE, 
                                     start_year, end_year, CHUNKS)

from downscaleml.main.inputoutput import (ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH)

from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
                                        ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
                                        ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
                                        PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES)

from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
from downscaleml.core.logging import log_conf
    
# module level logger
LOGGER = logging.getLogger(__name__)

def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape of the {} is in (spatial, time, variables):{}'.format(xarray_dataset, xarray_dataset.shape))
    total_elements = np.prod(xarray_dataset.shape)
    array_created = xarray_dataset.reshape((total_elements // xarray_dataset.shape[-1], xarray_dataset.shape[-1]))
    LogConfig.init_log('Reshaped to ndim=2 --> {} is in (space_time, variables):{}'.format(array_created, array_created.shape))
    return array_created

def doy_encoding(X, y=None, doy=False):

    # whether to include the day of the year as predictor variable
    if doy:
        # add doy to set of predictor variables
        LOGGER.info('Adding day of the year to predictor variables ...')
        X = X.assign(EoDataset.encode_doys(X, chunks=X.chunks))

    print(X)
    return X

def data_generator(X, y, chunk_size, shuffle=True):
    indices = list(range(len(X)))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(X), chunk_size):
        batch_indices = indices[i:i + chunk_size]
        yield X[batch_indices], y[batch_indices]



if __name__ == '__main__':

    # initialize timing
    start_time = time.monotonic()
        
    # initialize network filename
    state_file = NAMING_Model.state_file(
        NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, WET_DAY_THRESHOLD, dem=DEM,
        dem_features=DEM_FEATURES, doy=DOY, stratify=STRATIFY)
    
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
    target = target.joinpath(state_file.name + '.nc')
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
    Era5_ds = Era5_ds.rename({'lon': 'x','lat': 'y'})
    
    # initialize OBS predictand dataset
    LogConfig.init_log('Initializing observations for predictand: {}'
                       .format(PREDICTAND))

    # read in-situ gridded observations
    Obs_ds = search_files(OBS_PATH.joinpath(PREDICTAND), '.nc$').pop()
    Obs_ds = xr.open_dataset(Obs_ds)
    Obs_ds = Obs_ds.rename({'lon': 'x','lat': 'y'})

    # whether to use digital elevation model
    if DEM:
        # digital elevation model: Copernicus EU-Dem v1.1
        dem = search_files(DEM_PATH, '^interTwin_dem.nc$').pop()

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

    Era5_train = doy_encoding(Era5_train, Obs_train, doy=DOY)
    Era5_valid = doy_encoding(Era5_valid, Obs_valid, doy=DOY)

    predictors_train = Era5_train
    predictors_valid = Era5_valid
    predictand_train = Obs_train
    predictand_valid = Obs_valid
    pred_shape = Obs_valid
    
    predictors_train = stacker(predictors_train).compute()
    predictors_valid = stacker(predictors_valid).compute()
    predictand_train = stacker(predictand_train)
    predictand_valid = stacker(predictand_valid)
    
    stacked = pred_shape.stack()
    dask_arr = stacked.to_array().data
    pred_shape = dask_arr.T
    pred_shape = np.squeeze(pred_shape, axis=-1)
    LogConfig.init_log('Shape of the prediction is in (spatial, time, variables):{}'.format(pred_shape.shape))
    pred_shape = pred_shape.shape
    
    chunk_size = 10000
    
    LogConfig.init_log('Dask computations done!')
    # iterate over the grid points
    LogConfig.init_log('Downscaling by Random Forest Starts: iterating each grid cell over time dimension')
    
    Models = {
        'RandomForestRegressor' : RandomForestRegressor,
        'XGBRegressor' : XGBRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'LGBMRegressor': LGBMRegressor,
    }
    Model_name = NET

    #lgb_train = lgb.Dataset(predictors_train, predictand_train)  # Assuming 'X' and 'y' are the prepared data
    i = 0
    for chunk_X, chunk_y in data_generator(predictors_train, predictand_train, chunk_size=chunk_size):
        # Create LGBMDataset from the chunk
        lgb_chunk_dataset = lgb.Dataset(chunk_X, chunk_y)
        LogConfig.init_log('Chunk number : {:.2f}'.format(i))

        # Train the model on the current chunk
        model = lgb.train({}, lgb_chunk_dataset)
        i = i+1
       
    #prediction = np.ones(shape=(predictand_valid.shape[2], predictand_valid.shape[1], predictand_valid.shape[0])) * np.nan

    # instanciate the model for the current grid point
    #model = Models[Model_name]()

    # train model on training data
    #model.fit(predictors_train, predictand_train)
    # predict validation period
    prediction = model.predict(predictors_valid)
    LogConfig.init_log('Processing grid  - R2 score: {:.2f}'.format(r2_score(predictand_valid, prediction)))
    LogConfig.init_log('Processing grid  - MAE score: {:.2f}'.format(mean_absolute_error(predictand_valid, prediction)))
    LogConfig.init_log('Processing grid  - RMSE score: {:.2f}'.format(np.sqrt(mean_squared_error(predictand_valid, prediction))))

    # store predictions for current grid point
    # prediction[:, j, i] = pred

    LogConfig.init_log('Model ensemble saved and Indexed')
    prediction_reshaped = prediction.reshape(pred_shape)

    # store predictions in xarray.Dataset
    predictions = xr.DataArray(data=prediction_reshaped, dims=['x', 'y', 'time'],
                            coords=dict(lon=Obs_valid.x, lat=Obs_valid.y, time=pd.date_range(Era5_valid.time.values[0],Era5_valid.time.values[-1], freq='D')))
    predictions = predictions.to_dataset(name=PREDICTAND)

    predictions = predictions.set_index(
        x='lon', y='lat', time='time')
    
    # initialize network filename
    predict_file = NAMING_Model.state_file(
        NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, WET_DAY_THRESHOLD, dem=DEM,
        dem_features=DEM_FEATURES, doy=DOY, stratify=STRATIFY)

    predictions.to_netcdf("{}/{}.nc".format(str(target.parent), str(predict_file)))

    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE')
    
    
