import os
import pathlib
from ecmwfapi import ECMWFService
import numpy as np
from joblib import Parallel, delayed
import datetime

def dictionary_manipulator(date, config):
    if isinstance(date, str):
        date_value = int(date[:4])
        if date_value <= 2016:
            config['number'] = [str(n) for n in np.arange(0, 25)]
        else:
            config['number'] = [str(n) for n in np.arange(0, 51)]

    # Add additional keys based on the 'variable' key in config
    if levtype == "pressure_level":
        config['levtype'] = 'pl'
        config['level'] = level
        config['step'] = step = [str(s) for s in np.arange(0, 5172, 12)]
    elif levtype == "surface_level":
        config['levtype'] = 'sfc'
        config['step'] = step = [str(s) for s in np.arange(0, 5166, 6)]
    
    return config

# Constants
dates = []
param_dict = {
    "2m_temperature": "167.128",
    "total_precipitation": "228.128",
    "mean_sea_level_pressure": "151.128",
    "geopotential": "129.128",
    "specific_humity": "133.128",
    "temperature": "130.128",
    "u_component_of_wind": "131",
    "v_component_of_wind": "132" 
}
area = [52, 2, 40, 20]
step = [str(s) for s in np.arange(0, 5166, 6)]
level = [500, 850] # Incase you need to change the pressure levels

# Variables
levtype = "surface_level" # Change this to levtype = "surface_level"
if levtype == "pressure_level":
    param = ["129.128", "133.128", "130.128", "131", "132"]
elif levtype == "surface_level":
    param = ["167.128", "228.128", "151.128"]
SEAS5_PATH = pathlib.Path(f'/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/extra_SEAS5/{levtype}/')
years = [2017 ,2018, 2019]
for year in years:
    dates.extend([f"{year}{month:02d}01" for month in range(1, 13)])

# General Config Parameters to be passed
CONFIG = {
    "class": "od", #Iterable by years and months coded
    "expver": "1",
    "param": param,
    "method": "1",
    "origin": "ecmf",
    "stream": "mmsf",
    "system": "5",
    "time": "00:00:00",
    "type": "fc",
    "area": area,
    "grid": "0.4/0.4"
    }

OVERWRITE = False

if __name__ == '__main__':
    
    server = ECMWFService("mars")

    files = [SEAS5_PATH.joinpath('_'.join(['SEAS5', levtype, date]) + '.nc') for
                 date in dates]

    Parallel(n_jobs=2, verbose=51)(
        delayed(server.execute)(dictionary_manipulator(date, {**CONFIG, **{"date": date}}), file) 
        for file, date in zip(files, dates) if (not file.exists() or OVERWRITE))

