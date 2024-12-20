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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# locals
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset

from downscaleml.main.config import (ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH, SEAS5_PATH, NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                     CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                     OVERWRITE, DEM, DEM_FEATURES, STRATIFY,
                                     WET_DAY_THRESHOLD, VALID_SIZE, 
                                     start_year, end_year, CHUNKS, SEAS5_type, SEAS5_CHUNKS_forecast)

from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME, ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME, ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS, PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES)

from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
from downscaleml.core.logging import log_conf
    
# module level logger
LOGGER = logging.getLogger(__name__)

def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset

def dask_stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    #rechunker = (1, 161, 96, 365 ,15)
    #xarray_dataset = xarray_dataset.rechunk(rechunker)
    LogConfig.init_log('Shape is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset


def doy_encoding(X, y=None, doy=False):
    # whether to include the day of the year as predictor variable
    if doy:
        # add doy to set of predictor variables
        LOGGER.info('Adding day of the year to predictor variables ...')
        X = X.assign(EoDataset.encode_doys(X, chunks=X.chunks))

    return X

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
    Era5 = ERA5Dataset(ERA5_PATH.joinpath('ERA5_renamed'), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    Era5_ds = Era5.merge(chunks=CHUNKS)
    
    LogConfig.init_log('Initializing SEAS5 predictors.')

    #print(SEAS5_year)
    
    #TEMPORARY PATHWORK HERE - SOLVE IT FOR LATER
    Seas5 = ERA5Dataset(SEAS5_PATH.joinpath(f'{SEAS5_type}_SEAS5/'), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    Seas5_ds = Seas5.merge(chunks=SEAS5_CHUNKS_forecast)
    Seas5_ds = xr.unify_chunks(Seas5_ds)
    Seas5_ds = Seas5_ds[0]
    #Seas5_ds = Seas5_ds.rename({'longitude': 'x','latitude': 'y'})
    
    
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
        dem_path = search_files(DEM_PATH, '^interTwin_dem.nc$').pop()

        # read elevation and compute slope and aspect
        dem = ERA5Dataset.dem_features(
            dem_path, {'y': Era5_ds.y, 'x': Era5_ds.x},
            add_coord={'time': Era5_ds.time})
        
        dem_seas5 = ERA5Dataset.dem_features(
            dem_path, {'y': Seas5_ds.y, 'x': Seas5_ds.x},
            add_coord={'time': Seas5_ds.time})

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            dem = dem.drop_vars(['slope', 'aspect']).chunk(Era5_ds.chunks)
            dem_seas5_n = dem_seas5.merge(dem_seas5.expand_dims(number=Seas5_ds['number'])).chunk(Seas5_ds.chunks)
            dem_seas5_n = dem_seas5_n.drop_vars(['slope', 'aspect']).chunk(Seas5_ds.chunks)


        # add dem to set of predictor variables
        dem = dem.chunk(Era5_ds.chunks)
        dem_seas5_n = dem_seas5_n.chunk(Seas5_ds.chunks)
        Era5_ds = xr.merge([Era5_ds, dem])
        Seas5_ds = xr.merge([Seas5_ds, dem_seas5_n])

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
    Seas5_ds, Obs_valid = Seas5_ds.sel(time=VALID_PERIOD), Obs_ds.sel(time=VALID_PERIOD)
    dem_seas5, Obs_valid = dem_seas5.sel(time=VALID_PERIOD), Obs_ds.sel(time=VALID_PERIOD)

    Era5_train = doy_encoding(Era5_train, Obs_train, doy=DOY)
    dem_seas5 = doy_encoding(dem_seas5, Obs_valid, doy=DOY)

    Seas5_ds = Seas5_ds.merge(dem_seas5.expand_dims(number=Seas5_ds['number'])).chunk(Seas5_ds.chunks)
    Seas5_ds = Seas5_ds.drop_vars(['slope', 'aspect'])
    Seas5_ds = Seas5_ds.transpose('time', 'y', 'x', 'number')
    
    LogConfig.init_log('Era5_Train')
    print(Era5_train)
    LogConfig.init_log('SEAS5_Training Data Here')
    print(Seas5_ds)

    predictors_train = Era5_train
    predictors_valid = Seas5_ds
    predictand_train = Obs_train
    predictand_valid = Obs_valid

    #predictors_valid = dask_stacker(predictors_valid)
    predictors_train = stacker(predictors_train).compute()
    predictand_train = stacker(predictand_train)
    predictand_valid = stacker(predictand_valid)
    
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

    Era5_ds = None

    LogConfig.init_log('predictand_valid shape[0] : {}'.format(predictand_valid.shape[0]))
    LogConfig.init_log('predictand_valid shape[1] : {}'.format(predictand_valid.shape[1]))
    LogConfig.init_log('predictand_valid shape[2] : {}'.format(predictand_valid.shape[2]))

    LogConfig.init_log('predictors_train shape[0] : {}'.format(predictors_train.shape[0]))
    LogConfig.init_log('predictors_train shape[1] : {}'.format(predictors_train.shape[1]))
    LogConfig.init_log('predictors_train shape[2] : {}'.format(predictors_train.shape[2]))
    LogConfig.init_log('predictors_valid shape[3] : {}'.format(predictors_train.shape[3]))

    #LogConfig.init_log('predictors_valid shape[0] : {}'.format(predictors_valid.shape[0]))
    #LogConfig.init_log('predictors_valid shape[1] : {}'.format(predictors_valid.shape[1]))
    #LogConfig.init_log('predictors_valid shape[2] : {}'.format(predictors_valid.shape[2]))
    #LogConfig.init_log('predictors_valid shape[3] : {}'.format(predictors_valid.shape[3]))
    #LogConfig.init_log('predictors_valid shape[3] : {}'.format(predictors_valid.shape[4]))

    
    #for m in range(len(predictors_valid["number"])):
    for m in range(48, 51):
        
        prediction = np.ones(shape=(predictand_valid.shape[2], predictand_valid.shape[1], predictand_valid.shape[0])) * np.nan
        point_valid = stacker(predictors_valid.isel(number=m)).compute()
        #point_valid = point_valid[:, :, :, :]
        
        for i in range(predictors_train.shape[0]):
            for j in range(predictors_train.shape[1]):

                point_predictors = predictors_train[i, j, :, :]
                point_predictors = normalize(point_predictors)
                point_predictand = predictand_train[i, j, :, :]

                # check if the grid point is valid
                if np.isnan(point_predictors).any() or np.isnan(point_predictand).any():
                    # move on to next grid point
                    continue

                # prepare predictors of validation period
                point_validation = point_valid[i, j, :, :]
                point_validation = normalize(point_validation)

                predictand_validation = predictand_valid[i, j, :, :]

                LogConfig.init_log('Current grid point: ({:d}), ({:d}), ({:d}) '.format(m, i, j))    
                # normalize each predictor variable to [0, 1]
                # point_predictors = normalize(point_predictors)

                # instanciate the model for the current grid point
                model = Models[Model_name]()

                # train model on training data
                model.fit(point_predictors, point_predictand)
                # predict validation period
                pred = model.predict(point_validation)
                LogConfig.init_log('Processing grid point: {:d}, {:d}, {:d} - score: {:.2f}'.format(m, i, j, r2_score(predictand_validation, pred)))

                # store predictions for current grid point
                prediction[:, j, i] = pred

        LogConfig.init_log('Model ensemble for ensemble member {} saved and Indexed'.format(str(m)))

        # store predictions in xarray.Dataset
        predictions = xr.DataArray(data=prediction, dims=['time', 'y', 'x'],
                                coords=dict(time=pd.date_range(Seas5_ds.time.values[0],Seas5_ds.time.values[-1], freq='D'),
                                            lat=Obs_valid.y, lon=Obs_valid.x))
        predictions = predictions.to_dataset(name=PREDICTAND)

        predictions = predictions.set_index(
            time='time',
            y='lat',
            x='lon'
        )
        
        # initialize network filename
        predict_file = NAMING_Model.state_file(
            NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, WET_DAY_THRESHOLD, dem=DEM,
            dem_features=DEM_FEATURES, doy=DOY, stratify=STRATIFY)

        predictions.to_netcdf("{}/{}_{}_{}.nc".format(str(target.parent), str(predict_file), SEAS5_type, str(m)))

    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE!!')
    
    
