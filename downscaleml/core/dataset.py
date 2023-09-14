"""Dataset classes compliant to the Pytorch standard."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import logging
import pathlib
import warnings
from datetime import date

# externals
from osgeo import gdal
import torch
import numpy as np
import xarray as xr
import dask.array as da

# locals
from downscaleml.core.constants import (ERA5_VARIABLES, ERA5_PRESSURE_LEVELS,
                                        ERA5_P_VARIABLE_NAME, ERA5_S_VARIABLE_NAME,
                                        PROJECTION)
from downscaleml.core.utils import search_files, img2np, LogConfig

# module level logger
LOGGER = logging.getLogger(__name__)


class EoDataset(torch.utils.data.Dataset):

    @staticmethod
    def to_tensor(x, dtype=torch.float32):
        """Convert ``x`` to :py:class:`torch.Tensor`.

        Parameters
        ----------
        x : array_like
            The input data.
        dtype : :py:class:`torch.dtype`
            The data type used to convert ``x``.

            The modified class labels.
        Returns
        -------
        x : `torch.Tensor`
            The input data tensor.

        """
        return torch.tensor(x, dtype=dtype)

    @staticmethod
    def encode_cyclical_features(feature, max_val):
        """Encode a cyclical feature to the range [-1, 1].

        Parameters
        ----------
        feature : :py:class:`numpy.ndarray`
            The cyclcical feature to encode.
        max_val : `float`
            Maximum physically possible value of ``feature``.

        Returns
        -------
        encoded : `tuple` [:py:class:`numpy.ndarray`]
            The encoded feature in the range [-1, 1].

        """
        return (da.sin(2 * np.pi * feature / max_val).astype(np.float32),
                da.cos(2 * np.pi * feature / max_val).astype(np.float32))

    @staticmethod
    def add_coordinates(array, dims=('time', 'y', 'x')):
        return (dims, array)

    @staticmethod
    def repeat_along_axis(array, repeats, axis):
        return da.repeat(da.array(array), repeats, axis)

    @staticmethod
    def encode_doys(ds, dims=('time', 'y', 'x'), chunks=None):

        # compute day of the year
        LOGGER.info('Encoding day of the year to cyclical feature ...')
        doys = ds.time.values.astype('datetime64[D]')
        doys = da.asarray(
            [date.timetuple(doy.astype(date)).tm_yday for doy in doys])

        # reshape doys to correct shape: from (t,) to (t, y, x)
        # this expands the doy values to each pixel (y, x)
        target = (len(doys), len(ds.y), len(ds.x))
        repeat = int(target[-1] * target[-2])

        # encode day of the year as cyclical feature: convert to dask array
        sin_doy, cos_doy = EoDataset.encode_cyclical_features(doys, 365)

        # lazily repeat encoded doys along time
        sin_doy, cos_doy = (
            EoDataset.repeat_along_axis(sin_doy, repeat, 0).reshape(target),
            EoDataset.repeat_along_axis(cos_doy, repeat, 0).reshape(target))

        # chunk data for parallel loading
        if chunks is not None:
            sin_doy = sin_doy.rechunk(
                {dims.index(k): v for k, v in chunks.items()})
            cos_doy = cos_doy.rechunk(
                {dims.index(k): v for k, v in chunks.items()})

        return {'sin_doy': EoDataset.add_coordinates(sin_doy, dims),
                'cos_doy': EoDataset.add_coordinates(cos_doy, dims)}


    @staticmethod
    def dem_features(dem, coords, add_coord=None):

        LogConfig.init_log('Reading digital elevation model: {}'.format(dem))

        # compute slope
        LOGGER.info('Computing terrain slope ...')
        slope = gdal.DEMProcessing(
            '', str(dem), 'slope', format='MEM', computeEdges=True, alg='Horn',
            slopeFormat='degrees').ReadAsArray().astype(np.float32)

        # compute aspect
        LOGGER.info('Computing terrain aspect ...')
        aspect = gdal.DEMProcessing(
            '', str(dem), 'aspect', format='MEM', computeEdges=True,
            alg='Horn').ReadAsArray().astype(np.float32)

        # read digital elevation model
        dem = da.asarray(img2np(dem).astype(np.float32))
        slope = da.asarray(slope)
        aspect = da.asarray(aspect)

        # digital elevation model features: elevation, slope and aspect
        dem_vars = {
            'elevation': EoDataset.add_coordinates(dem, ('y', 'x')),
            'slope': EoDataset.add_coordinates(slope, ('y', 'x')),
            'aspect': EoDataset.add_coordinates(aspect, ('y', 'x'))}

        # create xarray.Dataset for digital elevation model
        dem_features = xr.Dataset(data_vars=dem_vars, coords=coords)

        # check whether to add additional coordinates
        if add_coord is not None:
            for var in dem_features.data_vars:
                # expand variable alogn new dimension
                for k, v in add_coord.items():
                    expanded = EoDataset.repeat_along_axis(
                        np.expand_dims(dem_features[var], axis=0),
                        len(v), axis=0)

                    # overwrite variable
                    dem_features[var] = EoDataset.add_coordinates(
                        expanded, (k, 'y', 'x'))

            # assign new coordinate
            dem_features = dem_features.assign_coords(add_coord)

        return dem_features

    @staticmethod
    def anomalies(ds, timescale='time.dayofyear', standard=False):
        # group dataset by time scale
        LOGGER.info('Computing anomalies ...')
        groups = ds.groupby(timescale).groups

        # compute anomalies over time
        anomalies = {}
        for time, time_scale in groups.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                # anomaly = (x(t) - mean(x, t))
                anomalies[time] = (ds.isel(time=time_scale) -
                                   ds.isel(time=time_scale).mean(dim='time'))

                # standardized anomaly = (x(t) - mean(x, t)) / std(x, t)
                if standard:
                    anomalies[time] /= ds.isel(time=time_scale).std(dim='time')

        # concatenate anomalies and sort chronologically
        anomalies = xr.concat(anomalies.values(), dim='time')
        anomalies = anomalies.sortby(anomalies.time)

        return anomalies

    @staticmethod
    def normalize(ds, dim=('time', 'y', 'x'), period=None):
        # normalize predictors to [0, 1]
        LOGGER.info('Normalizing data to [0, 1] ...')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)

            # whether to normalize using statistics for a specific period
            # NOTE: this can result in values that are not in [0, 1]
            if period is not None:
                ds -= ds.sel(time=period).min(dim=dim)
                ds /= ds.sel(time=period).max(dim=dim)
            # normalize using entire period: [0, 1]
            else:
                ds -= ds.min(dim=dim)
                ds /= ds.max(dim=dim)

        return ds


class NetCDFDataset(EoDataset):

    def __init__(self, X, y=None, normalize=True, doy=False, anomalies=False):

        # whether to include the day of the year as predictor variable
        if doy:
            # add doy to set of predictor variables
            LOGGER.info('Adding day of the year to predictor variables ...')
            X = X.assign(self.encode_doys(X, chunks=X.chunks))

        print(X)
        # NetCDF dataset containing predictor variables (ERA5)
        # shape: (t, vars, y, x)
        self.X = X.to_array().values.swapaxes(0, 1)

        # whether to train on standardized anomalies over time
        if anomalies:
            self.X = ((self.X - self.X.mean(axis=(0, -1, -2), keepdims=True)) /
                      self.X.std(axis=(0, -1, -2), keepdims=True))

        # whether to normalize the training data to [0, 1] or standardize to
        # mean=0, std=1
        if normalize:
            # min-max-scaling: x in [0, 1]
            LOGGER.info('Normalizing data to [0, 1] ...')
            self.X -= self.X.min(axis=(0, -1, -2), keepdims=True)
            self.X /= self.X.max(axis=(0, -1, -2), keepdims=True)
        else:
            # standard scaling: mean=0, std=1
            LOGGER.info('Standardizing data to mean=0, std=1 ...')
            mean = np.nanmean(self.X, axis=(0, -1, -2), keepdims=True)
            std = np.nanstd(self.X, axis=(0, -1, -2), keepdims=True, ddof=1)
            self.X = (self.X - mean) / std

        # convert predictors to torch.Tensor
        self.X = self.to_tensor(self.X)

        # NetCDF dataset containing target variable (OBS)
        self.y = torch.zeros_like(self.X)
        if y is not None:
            self.y = self.to_tensor(y.to_array().values.swapaxes(0, 1))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # return a training sample given an index
        return self.X[idx, ...], self.y[idx, ...]


class ERA5Dataset(EoDataset):

    # compression for NetCDF
    comp = dict(dtype='float32', complevel=5, zlib=True)

    def __init__(self, root_dir, variables, plevels):

        # root directory: search for ERA5 files
        self.root_dir = pathlib.Path(root_dir)

        # valid variable names
        self.variables = [var for var in variables if var in ERA5_VARIABLES]

        # pressure levels
        self.plevels = [pl for pl in plevels if pl in ERA5_PRESSURE_LEVELS]

    def merge(self, **kwargs):
        # search dataset for each variable in root directory
        datasets = [search_files(self.root_dir.joinpath(var), '.nc$').pop() for
                    var in self.variables]

        # iterate over input datasets and select pressure levels
        LOGGER.info('Variables: {}'.format(', '.join(self.variables)))
        LOGGER.info('Pressure levels (hPa): {}.'.format(
            ', '.join([str(pl) for pl in self.plevels])))
        predictors = []
        for ds in datasets:
            # read dataset
            ds = xr.open_dataset(ds, **kwargs)

            # check if the dataset is defined on pressure levels or single
            # levels
            if 'level' in ds.dims:
                # iterate over pressure levels to use
                for pl in self.plevels:
                    # check if pressure level is available
                    if pl not in ds.level:
                        LOGGER.info('Pressure level {:d} hPa not available.'
                                    .format(pl))
                        continue

                    # select pressure level and drop unnecessary dimensions
                    level = ds.sel(level=pl).drop('level')

                    # rename variable including corresponding pressure level
                    level = level.rename({k: '_'.join([k, str(pl)]) for k in
                                          level.data_vars if k != PROJECTION})

                    # append current pressure level to list of predictors
                    predictors.append(level)
            else:
                # append single level dataset to list of predictors
                predictors.append(ds)

        # drop projection variable, if present
        # predictors = [ds.drop_vars(PROJECTION) for ds in predictors if
                     # PROJECTION in ds.data_vars]

        # merge final predictor dataset
        return xr.merge(predictors)
    
