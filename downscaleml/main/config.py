import datetime
import numpy as np
import pathlib


from downscaleml.core.constants import (PREDICTANDS, ERA5_P_VARIABLES,
                                   ERA5_S_VARIABLES, MODELS)

PREDICTAND='pr'
assert PREDICTAND in PREDICTANDS

NET='LGBMRegressor'
assert NET in MODELS

if PREDICTAND is 'pr':
    ERA5="p_REANALYSIS"
else:
    ERA5="REANALYSIS"

ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/hydroModelDownscale/')

# path to this file
HERE = pathlib.Path(__file__).parent

ERA5_PATH = ROOT.joinpath(ERA5)

SEAS5_PATH = ROOT

OBS_PATH = ROOT.joinpath('CERRA')

DEM_PATH = ROOT.joinpath('DEM')

RESULTS = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/sf_downscaling/')

MODEL_PATH = RESULTS.joinpath('RESULTS_model')

TARGET_PATH = RESULTS.joinpath('RESULTS')

# SEAS5 hindcast or forecast

SEAS5_type = "forecast"
# SEAS5_type = "forecast"

# include day of year as predictor
DOY = True

# whether to normalize the training data to [0, 1]
NORM = True

# whether to overwrite existing models
OVERWRITE = False

# whether to use DEM slope and aspect as predictors
DEM_FEATURES = False

# stratify training/validation set for precipitation by number of wet days
STRATIFY = True

# size of the validation set w.r.t. the training set
# e.g., VALID_SIZE = 0.2 means: 80% of CALIB_PERIOD for training
#                               20% of CALIB_PERIOD for validation
VALID_SIZE = 0.2

CHUNKS = {'time': 365, 'x': 161, 'y': 96}
#SEAS5_CHUNKS_forecast = {'time': 1429, 'x': 161, 'y': 96, 'number': 1}
SEAS5_CHUNKS_forecast = {'number': 1, 'x': 161, 'y': 96, 'time': 365}
SEAS5_CHUNKS = {'time': 365, 'x': 161, 'y': 96, 'number': 1}

# threshold  defining the minimum amount of precipitation (mm) for a wet day
WET_DAY_THRESHOLD=0

if PREDICTAND is 'tasmean':
    ERA5_P_PREDICTORS = ['geopotential', 'temperature', 'u_component_of_wind',
                          'v_component_of_wind', 'specific_humidity']
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["mean_sea_level_pressure", "2m_temperature"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])

elif PREDICTAND is 'tasmin':
    ERA5_P_PREDICTORS = ['geopotential', 'temperature_min', 'u_component_of_wind',
                          'v_component_of_wind', 'specific_humidity']
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["mean_sea_level_pressure", "2m_tasmin"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])

elif PREDICTAND is 'tasmax':
    ERA5_P_PREDICTORS = ['geopotential', 'temperature_max', 'u_component_of_wind',
                          'v_component_of_wind', 'specific_humidity']
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["mean_sea_level_pressure", "2m_tasmax"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])

else:
    ERA5_P_PREDICTORS = ['geopotential', 'temperature', 'u_component_of_wind',
                          'v_component_of_wind', 'specific_humidity']
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])


# ERA5 predictor variables
ERA5_PREDICTORS = ERA5_P_PREDICTORS + ERA5_S_PREDICTORS

# ERA5 pressure levels
ERA5_PLEVELS = [500, 850]

DEM = True
if DEM:
    # remove model orography when using DEM
    if 'orography' in ERA5_S_PREDICTORS:
        ERA5_S_PREDICTORS.remove('orography')

#NET='AdaBoostRegressor'
#NET='AdaBoostRegressor'
#NET='AdaBoostRegressor'
#NET='AdaBoostRegressor'

CALIB_PERIOD = np.arange(
    datetime.datetime.strptime('1985-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2016-02-01', '%Y-%m-%d').date())

start_year = np.min(CALIB_PERIOD).astype(datetime.datetime).year
end_year = np.max(CALIB_PERIOD).astype(datetime.datetime).year

# validation period: testing
VALID_PERIOD = np.arange(
    datetime.datetime.strptime('2017-02-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date())

SEAS5_year = np.min(VALID_PERIOD).astype(datetime.datetime).year

combination = 1

params = {'reg_alpha': 0.001018337175296235,
 'reg_lambda': 0.10078524608920145,
 'colsample_bytree': 0.7,
 'subsample': 0.8,
 'learning_rate': 0.01,
 'max_depth': 100,
 'num_leaves': 526,
 'min_child_samples': 59,
 'random_state': 48,
 'n_estimators': 20000,
 'metric': 'rmse',
 'cat_smooth': 91}

paramss = {'reg_alpha': 0.008623170303712244, 
           'reg_lambda': 0.0012155495918039461, 
           'colsample_bytree': 1.0, 
           'subsample': 1.0, 
           'learning_rate': 0.008, 
           'max_depth': 20, 
           'num_leaves': 454, 
           'min_child_samples': 54,
           'random_state': 48,
           'n_estimators': 20000,
           'metric': 'rmse',
           'cat_smooth': 91}
