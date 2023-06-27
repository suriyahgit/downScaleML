import xarray as xr
import numpy as np
import subprocess

variable = "total_precipitation"

path = "/mnt/CEPH_PROJECTS/FACT_CLIMAX/REANALYSIS/Downloads/{}/*.nc".format(variable, variable)
outpath = "/home/sdhinakaran/ERA5_Daily_Precipi/{}/ERA5_{}_1985_to_2016_daily_time_shifted.nc".format(variable, variable)
regridded = "/home/sdhinakaran/ERA5_Daily_Precipi/{}/reproj_ERA5_{}_1985_to_2016_daily_time_shifted.nc".format(variable, variable)

ds = xr.open_mfdataset(path, concat_dim='time', combine='nested')
ds = ds.shift(time = -6)
ds = ds.resample(time='D').mean(dim='time')
ds = ds.drop([np.datetime64('2016-12-31')], dim='time')
ds.to_netcdf(outpath)
print(ds)

subprocess.call(["cdo", "remapbil,grid_ext", outpath, regridded])
ds = xr.open_dataset(regridded)
if 'spatial_ref' in ds.variables:
    ds = ds.drop_vars("spatial_ref")
print(ds)

item = "/home/sdhinakaran/ERA5_Daily_Precipi/{}/ERA5_{}_1985_to_2016_daily_p.nc".format(variable, variable)
comp = dict(dtype='float32', zlib=True, complevel=5)
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(item, engine = "h5netcdf", encoding=encoding)



