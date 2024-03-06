import numpy as np
import enum

""" Config constants """

# ERA5 predictor variables on pressure levels
ERA5_P_VARIABLES = ['geopotential', 'temperature', 'u_component_of_wind',
                    'v_component_of_wind', 'specific_humidity', 'temperature_max', 'temperature_min']
ERA5_P_VARIABLES_SHORTCUT = ['z', 't', 'u', 'v', 'q', 't', 't']
ERA5_P_VARIABLE_NAME = {k: v for k, v in zip(ERA5_P_VARIABLES,
                                             ERA5_P_VARIABLES_SHORTCUT)}

# ERA5 predictor variables on single levels
ERA5_S_VARIABLES = ['mean_sea_level_pressure', '2m_temperature',
                    'total_precipitation', 'surface_pressure',
                    '2m_dewpoint_temperature', '2m_tasmax', '2m_tasmin']
ERA5_S_VARIABLES_SHORTCUT = ['mslp', 't2m', 'pr', 'p', 't2d', 't2m', 't2m']
ERA5_S_VARIABLE_NAME = {k: v for k, v in zip(ERA5_S_VARIABLES,
                                             ERA5_S_VARIABLES_SHORTCUT)}

# ERA5 predictor variables
ERA5_VARIABLES = ERA5_P_VARIABLES + ERA5_S_VARIABLES

# ERA5 predictor variables: mapping from long name to short cut
ERA5_VARIABLE_NAMES = {**ERA5_P_VARIABLE_NAME, **ERA5_S_VARIABLE_NAME}

# ERA5 pressure levels
ERA5_PRESSURE_LEVELS = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750,
                        700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225,
                        200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3,
                        2, 1]

PREDICTANDS = ['tasmin', 'tasmax', 'pr', 'tasmean']

#ERA5_P_VARIABLES = ['geopotential', 'temperature', 'u_component_of_wind',
#                    'v_component_of_wind', 'specific_humidity', 'temperature_max', 'temperature_min']

#ERA5_S_VARIABLES = ['mean_sea_level_pressure', '2m_temperature',
#                    'total_precipitation', 'surface_pressure',
#                    '2m_dewpoint_temperature', 'dem_1km', '2m_tasmax', '2m_tasmin']

PROJECTION = 'lambert_azimuthal_equal_area'

# climate data operator (cdo) resampling modes
CDO_RESAMPLING_MODES = ['bilinear', 'conservative']

YEARS = list(range(1981, 2021))

CERRA_VARIABLES = ["total_precipitation", "2m_temperature", "2m_tasmin", "2m_tasmin"]

MODELS = ["RandomForestRegressor", "XGBRegressor", "AdaBoostRegressor", "LGBMRegressor"]


class Gdal2Numpy(enum.Enum):
    """Data type mapping from gdal to numpy."""

    Byte = np.uint8
    UInt8 = np.uint8
    Int8 = np.int8
    UInt16 = np.uint16
    Int16 = np.int16
    UInt32 = np.uint32
    Int32 = np.int32
    Float32 = np.float32
    Float64 = np.float64
    CInt16 = np.complex64
    CInt32 = np.complex64
    CFloat32 = np.complex64
    CFloat64 = np.complex64