import os
import pathlib
from ecmwfapi import ECMWFService
import numpy as np
from joblib import Parallel, delayed

# Constants
twom_temperature = "167.128"
total_precipitation = "228.128"
mean_sea_level_pressure = "151.128"
area = [52, 2, 40, 20]
number = [str(n) for n in np.arange(0, 25)]
step = [str(s) for s in np.arange(0, 5166, 6)]
dates1 = ["20200101", "20200201", "20200401", "20200501", "20200601", "20200701","20200801","20200901","20201001", "20201101", "20201201", "20160101", "20160201", "20160301", "20160401", "20160501", "20160601", "20160701","20160801","20160901","20161001", "20161101", "20161201"]

dates = ["20160401", "20160501", "20160601", "20160701","20160801","20160901","20161001", "20161101", "20161201"]

# Variables

var = twom_temperature # Change this!
SEAS5_PATH = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/SEAS5/2m_temperature/') # Change this

if var == twom_temperature: 
    var_str = "2m_temperature"
elif var == total_precipitation:
    var_str = "total_precipitation"
else:
    var_str = "mean_sea_level_pressure"

# General Config Parameters to be passed

CONFIG = {
    "class": "od", #Iterable by years and months coded
    "expver": "1",
    "levtype": "sfc",
    "param": var,
    "method": "1",
    "number": number,
    "origin": "ecmf",
    "step": step,
    "stream": "mmsf",
    "system": "5",
    "time": "00:00:00",
    "type": "fc",
    "area": area,
    "grid": "0.4/0.4",
    "target": ""}

OVERWRITE = False

if __name__ == '__main__':
    
    server = ECMWFService("mars")

    files = [SEAS5_PATH.joinpath('_'.join(['SEAS5', var_str, date]) + '.grib') for
                 date in dates]

    Parallel(n_jobs=4, verbose=51)(
        delayed(server.execute)({**CONFIG, **{"date": date}}, file)
        for file, date in zip(files, dates) if (not file.exists() or OVERWRITE))

