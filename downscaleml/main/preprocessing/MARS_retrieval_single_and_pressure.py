import os
import pathlib
from ecmwfapi import ECMWFService
import numpy as np
from joblib import Parallel, delayed

def ensemble_number_check(date, config):
    if isinstance(date, str):
        date_value = datetime.strptime(date, '%Y-%m-%d').date()
        if date_value.year =< 2016:
            config['number'] = [str(n) for n in np.arange(0, 25)]
        else:
            config['number'] = [str(n) for n in np.arange(0, 51)]

    # Add additional keys based on the 'variable' key in config
    if var in ["2m_temperature", "mean_sea_level_pressure", "total_precipitation"]:
        config['levtype'] = 'sfc'
    elif var in ["u_component_of_wind", "v_component_of_wind", "geopotential", "temperature", "specific_humidity"]:
        config['levtype'] = 'pl'
        config['level'] = level
    
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
var = "2m_temperature" # Change this
param = param_dict[var] 
SEAS5_PATH = pathlib.Path(f'/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/SEAS5/{var}/')
years = [2016, 2020]
for year in years:
    dates.extend([f"{year}{month:02d}01" for month in range(1, 13)])

# General Config Parameters to be passed

CONFIG = {
    "class": "od", #Iterable by years and months coded
    "expver": "1",
    "param": param,
    "method": "1",
    "origin": "ecmf",
    "step": step,
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

    files = [SEAS5_PATH.joinpath('_'.join(['SEAS5', var, date]) + '.grib') for
                 date in dates]

    Parallel(n_jobs=4, verbose=51)(
        delayed(server.execute)(ensemble_number_check(date, {**CONFIG, **{"date": date}}), file) 
        for file, date in zip(files, dates) if (not file.exists() or OVERWRITE))

