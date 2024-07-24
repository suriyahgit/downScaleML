import datetime
import numpy as np
import pathlib


from downscaleml.core.constants import (PREDICTANDS, ERA5_P_VARIABLES,
                                   ERA5_S_VARIABLES, MODELS)

PREDICTAND='tasmax'
assert PREDICTAND in PREDICTANDS

NET='LGBMRegressor'
assert NET in MODELS

if PREDICTAND is 'pr':
    ERA5="p_REANALYSIS"
else:
    ERA5="HISTORICAL/r101i1p1f1/"

ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/cmip_downscale/')

# path to this file
HERE = pathlib.Path(__file__).parent

ERA5_PATH = ROOT.joinpath(ERA5)

SEAS5_PATH = ROOT

PROJECTION_PATH = ROOT.joinpath('PROJECTION')

OBS_PATH = ROOT.joinpath('MSWX')

DEM_PATH = ROOT.joinpath('DEM')

RESULTS = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/cmip_downscale/')

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
    ERA5_P_PREDICTORS = ['geopotential', 'temperature', 'u_component_of_wind',
                          'v_component_of_wind']
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["mean_sea_level_pressure", "tasmax"]
    
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
ERA5_PLEVELS = [250, 500, 850]

DEM = True
if DEM:
    # remove model orography when using DEM
    if 'orography' in ERA5_S_PREDICTORS:
        ERA5_S_PREDICTORS.remove('orography')


CALIB_PERIOD = np.arange(
    datetime.datetime.strptime('1979-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date())

start_year = np.min(CALIB_PERIOD).astype(datetime.datetime).year
end_year = np.max(CALIB_PERIOD).astype(datetime.datetime).year

# validation period: testing
VALID_PERIOD = np.arange(
    datetime.datetime.strptime('2015-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2100-12-31', '%Y-%m-%d').date())

SEAS5_year = np.min(VALID_PERIOD).astype(datetime.datetime).year
