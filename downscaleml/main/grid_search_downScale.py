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
import plotly
import optuna
from sklearn.metrics import mean_squared_error
import math
import random
from functools import partial

# externals
import xarray as xr

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# locals
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset

from downscaleml.main.config import (NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                     CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                     OVERWRITE, DEM, DEM_FEATURES, STRATIFY,
                                     WET_DAY_THRESHOLD, VALID_SIZE, 
                                     start_year, end_year, CHUNKS, combination, paramss)

from downscaleml.main.inputoutput import (ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH)

from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
                                        ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
                                        ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
                                        PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES)

from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
from downscaleml.core.logging import log_conf

from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as stats
from IPython.display import Image
from sklearn.metrics import r2_score
    
# module level logger
LOGGER = logging.getLogger(__name__)

def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape of the numpy array is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset

def doy_encoding(X, y=None, doy=False):

    # whether to include the day of the year as predictor variable
    if doy:
        # add doy to set of predictor variables
        LOGGER.info('Adding day of the year to predictor variables ...')
        X = X.assign(EoDataset.encode_doys(X, chunks=X.chunks))

    print(X)
    return X

random.seed(42)

def grid_creator(combinations, numpy_object):
    gridded = np.ones(shape=(int(math.sqrt(combinations)), int(math.sqrt(combinations)), numpy_object.shape[2], numpy_object.shape[3])) * np.nan
    return gridded

def generate_unique_random_array(size, lower_bound, upper_bound, seed=None):

    if size > upper_bound - lower_bound:
        raise ValueError("Cannot generate more unique values than the range size.")

    if seed is not None:
        random.seed(seed)

    unique_values = set()

    while len(unique_values) < size:
        unique_values.add(random.randint(lower_bound, upper_bound))

    # Convert the set to a list and return it
    return list(unique_values)

# Define a seed value (or choose any other value)
seed_value = 42

def objective(trial,data,target):

    #General parameters that would fit in any data
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,shuffle=False)
    param = {
        'metric': 'rmse', 
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100),
        'verbose' : -1,
        'early_stopping' : 100
    }

    model = LGBMRegressor()
    rmse_values = np.ones(shape=(int(math.sqrt(combination)), int(math.sqrt(combination))))
    
    for i in range(train_x.shape[1]):
        for j in range(train_x.shape[2]):

            point_predictors = train_x[:, j, i, :]
            point_predictand = train_y[:, j, i, :]
    
            # check if the grid point is valid
            if np.isnan(point_predictors).any() or np.isnan(point_predictand).any():
                # move on to next grid point
                continue

            point_validation = test_x[:, j, i, :]
            #point_validation = normalize(point_validation)

            predictand_validation = test_y[:, j, i, :]

            LogConfig.init_log('Current grid point: ({:d}), ({:d}) '.format(i, j))   
    
            model.fit(point_predictors,np.squeeze(point_predictand),eval_set=[(point_validation,predictand_validation)])

    for i in range(train_x.shape[1]):
        for j in range(train_x.shape[2]):
            # prepare predictors of validation period
            point_validation = test_x[:, j, i, :]
            #point_validation = normalize(point_validation)

            predictand_validation = test_y[:, j, i, :]
            
            preds = model.predict(point_validation)
            
            rmse = mean_squared_error(predictand_validation, preds,squared=False)
            rmse_values[i][j] = rmse

            LogConfig.init_log('Current grid point: ({:d}), ({:d}) RMSE: {} '.format(i, j, rmse_values[i][j]))    
            

    mean_rmse = np.mean(rmse_values)
    return mean_rmse


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
    target = target.joinpath(state_file.name + 'll.nc')
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
    
    predictors_train = stacker(predictors_train).compute()
    predictors_valid = stacker(predictors_valid).compute()
    predictand_train = stacker(predictand_train)
    predictand_valid = stacker(predictand_valid)
    
    LogConfig.init_log('Dask computations done!')
    
    Models = {
        'RandomForestRegressor' : RandomForestRegressor,
        'XGBRegressor' : XGBRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'LGBMRegressor': LGBMRegressor,
    }
    Model_name = NET

    predictors_train_grid = grid_creator(combination, predictors_train)
    predictand_train_grid = grid_creator(combination, predictand_train)
    predictors_valid_grid = grid_creator(combination, predictors_valid)
    predictand_valid_grid = grid_creator(combination, predictand_valid)

    x = generate_unique_random_array(10, 0, (predictand_train.shape[0] - 1), seed_value)
    y = generate_unique_random_array(10, 0, (predictand_train.shape[1] - 1), seed_value)
    
    LogConfig.init_log('The reduced grid lat and lon values here: {}, {}'.format(x, y))

    for i in range(int(math.sqrt(combination))):
        for j in range(int(math.sqrt(combination))):
            predictors_train_grid[i][j] = predictors_train[x[i], y[j], :, :]
            predictand_train_grid[i][j] = predictand_train[x[i], y[j], :, :]
            predictors_valid_grid[i][j] = predictors_valid[x[i], y[j], :, :] 
            predictand_valid_grid[i][j] = predictand_valid[x[i], y[j], :, :]

    LogConfig.init_log('Reduced grid extraction complete!')

    lon, lat = np.squeeze(Obs_valid.x.values), np.squeeze(Obs_valid.y.values)

    grid_x, grid_y = lon[x], lat[y]

    reduced_grid_x = xr.DataArray(grid_x, dims=('x'))
    reduced_grid_y = xr.DataArray(grid_y, dims=('y'))

    predictors_train_grid = predictors_train_grid.transpose(2, 1, 0, 3)
    predictand_train_grid = predictand_train_grid.transpose(2, 1, 0, 3)
    predictors_valid_grid = predictors_valid_grid.transpose(2, 1, 0, 3)
    predictand_valid_grid = predictand_valid_grid.transpose(2, 1, 0, 3)

    objective_with_args = partial(objective, data=predictors_train_grid,target=predictand_train_grid)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_with_args, n_trials=50)

    LogConfig.init_log('Number of finished trials: {}'.format(len(study.trials)))
    LogConfig.init_log('Best trial: {}'.format(study.best_trial.params))
    
    params=study.best_params
    params['random_state'] = 48
    params['n_estimators'] = 20000 
    params['metric'] = 'rmse'
    params['cat_smooth'] = params.pop('min_data_per_groups')
    LogConfig.init_log('Best Params: {} for the Predictand: {}'.format(params, PREDICTAND))
    LogConfig.init_log("Best Hyperparameter Pokemon Captured!")
    


    









