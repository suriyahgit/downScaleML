import datetime
import numpy as np

from downscaleml.core.constants import (PREDICTANDS, ERA5_P_VARIABLES,
                                   ERA5_S_VARIABLES, MODELS)

# builtins

# include day of year as predictor
DOY = True

# whether to normalize the training data to [0, 1]
NORM = True

# whether to overwrite existing models
OVERWRITE = False

# whether to use DEM slope and aspect as predictors
DEM_FEATURES = True

# stratify training/validation set for precipitation by number of wet days
STRATIFY = True

# size of the validation set w.r.t. the training set
# e.g., VALID_SIZE = 0.2 means: 80% of CALIB_PERIOD for training
#                               20% of CALIB_PERIOD for validation
VALID_SIZE = 0.2

CHUNKS = {'time': 365}

# threshold  defining the minimum amount of precipitation (mm) for a wet day
WET_DAY_THRESHOLD=0


ERA5_P_PREDICTORS = ['geopotential', 'temperature', 'u_component_of_wind',
                      'v_component_of_wind', 'specific_humidity']
                      
#ERA5_P_PREDICTORS = []
assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])

# ERA5 predictor variables on single levels
ERA5_S_PREDICTORS=["mean_sea_level_pressure", "2m_temperature"]

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

NET='LGBMRegressor'
assert NET in MODELS

PREDICTAND='tasmean'
assert PREDICTAND in PREDICTANDS

CALIB_PERIOD = np.arange(
    datetime.datetime.strptime('1985-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2016-01-01', '%Y-%m-%d').date())

start_year = np.min(CALIB_PERIOD).astype(datetime.datetime).year
end_year = np.max(CALIB_PERIOD).astype(datetime.datetime).year

# validation period: testing
VALID_PERIOD = np.arange(
    datetime.datetime.strptime('2016-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2020-12-31', '%Y-%m-%d').date())

combination = 100

