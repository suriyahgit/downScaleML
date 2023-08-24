import cdo
import numpy as np
import pathlib
import pandas as pd
import logging
import re
import os
import datetime
from logging.config import dictConfig
import dataclasses
from osgeo import gdal, ogr, osr
import enum

from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
                                        ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
                                        ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
                                        PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES, Gdal2Numpy, CDO_RESAMPLING_MODES)

# module level logger
LOGGER = logging.getLogger(__name__)

def reproject_cdo(grid, src_ds, trg_ds, mode='bilinear', overwrite=False):

    # instanciate the cdo
    operator = cdo.Cdo()

    # check if dataset exists
    trg_ds = pathlib.Path(trg_ds)
    if trg_ds.exists() and not overwrite:
        LOGGER.info('{} already exists. Aborting ...'.format(trg_ds))
        return trg_ds

    # check if mode is supported
    if mode not in CDO_RESAMPLING_MODES:
        raise ValueError('Resampling mode "{}" not supported.'.format(mode))
    else:
        # check which resampling mode to use
        LOGGER.info('Reproject: {}'.format(trg_ds))
        if mode == 'bilinear':
            operator.remapbil(str(grid),
                              input=str(src_ds), output=str(trg_ds))

        if mode == 'conservative':
            operator.remapcon(str(grid),
                              input=str(src_ds), output=str(trg_ds))

    return trg_ds


def normalize(predictors):
    predictors -= predictors.min(axis=1, keepdims=True)
    predictors /= predictors.max(axis=1, keepdims=True)
    return predictors

def rename_file(old_path, new_path):
    try:
        os.rename(old_path, new_path)
        LOGGER.info(f"File '{old_path}' renamed to '{new_path}' successfully.")
    except FileNotFoundError:
        LOGGER.info(f"Error: File '{old_path}' not found.")
    except FileExistsError:
        LOGGER.info(f"Error: File '{new_path}' already exists.")

class NAMING_Model:
    @staticmethod
    def state_file(NET, predictand, predictors, plevels, wet_day, dem=False,
                   dem_features=False, doy=False, stratify=False):
        
        Ppredictors = ''.join([ERA5_P_VARIABLE_NAME[p] for p in predictors if p
                               in ERA5_P_VARIABLE_NAME])
        Spredictors = ''.join([ERA5_S_VARIABLE_NAME[p] for p in predictors if p
                               in ERA5_S_VARIABLE_NAME])
        plevels = [str(p) for p in plevels]
        state_file = '_'.join([NET, str(predictand)])

        # check if predictors on pressure levels are used
        state_file = ('_'.join([state_file, Ppredictors, *plevels]) if
                      any([p in ERA5_P_VARIABLE_NAME for p in predictors]) else
                      state_file)

        # check if predictors on surface are used
        state_file = ('_'.join([state_file, Spredictors]) if
                      any([p in ERA5_S_VARIABLE_NAME for p in predictors]) else
                      state_file)
        
        state_file = '_'.join([state_file, str(wet_day)]) if stratify else state_file
        state_file = ''.join([state_file,'mm']) if stratify else state_file

        # check whether digital elevation model, slope and aspect, and the day
        # of year were used
        state_file = '_'.join([state_file, 'dem']) if dem else state_file
        state_file = ('_'.join([state_file, 'sa']) if dem_features else
                      state_file)
        state_file = '_'.join([state_file, 'doy']) if doy else state_file
            
        return state_file
    
def search_files(directory, pattern):
    """Recursively searches files matching a pattern.

    Parameters
    ----------
    directory : `str` or :py:class:`pathlib.Path`
        The directory to recursively search.
    pattern : `str`
        The pattern to match. Regular expressions are supported.

    Returns
    -------
    matches : `list` [:py:class:`pathlib.Path`]
        List of files in ``directory`` matching ``pattern``.

    """
    LOGGER.info('Searching: {}, pattern: {}'.format(directory, pattern))

    # create regular expression
    pattern = re.compile(pattern)

    # recursively search for files matching the pattern
    matches = []
    for dirpath, _, files in os.walk(directory):
        matches.extend([pathlib.Path(dirpath).joinpath(file) for file in files
                        if pattern.search(file)])

    return matches


@dataclasses.dataclass
class BaseConfig:
    """Base :py:class:`dataclasses.dataclass` for each configuration."""

    def __post_init__(self):
        """Check the type of each argument.

        Raises
        ------
        TypeError
            Raised if the conversion to the specified type of the argument
            fails.

        """
        # check input types
        for field in dataclasses.fields(self):
            # the value of the current field
            value = getattr(self, field.name)

            # check whether the value is of the correct type
            if not isinstance(value, field.type):
                # try to convert the value to the correct type
                try:
                    setattr(self, field.name, field.type(value))
                except TypeError:
                    # raise an exception if the conversion fails
                    raise TypeError('Expected {} to be {}, got {}.'
                                    .format(field.name, field.type,
                                            type(value)))


@dataclasses.dataclass
class LogConfig(BaseConfig):
    """Logging configuration class.

    Generate the model log file.

    Attributes
    ----------
    state_file : :py:class:`pathlib.Path`
        Path to a model state file.
    log_path : :py:class:`pathlib.Path`
        Path to store model logs.
    log_file : :py:class:`pathlib.Path`
        Path to the log file of the model ``state_file``.
    """

    state_file: pathlib.Path

    def __post_init__(self):
        """Check the type of each argument.

        Generate model log file.

        """
        super().__post_init__()

        # the path to store model logs
        self.log_path = pathlib.Path(HERE).joinpath('_logs')

        # the log file of the current model
        self.log_file = check_filename_length(self.log_path.joinpath(
            self.state_file.name.replace('.pt', '.log')))

    @staticmethod
    def now():
        """Return the current date and time.

        Returns
        -------
        date : :py:class:`datetime.datetime`
            The current date and time.

        """
        return datetime.datetime.strftime(datetime.datetime.now(),
                                          '%Y-%m-%dT%H:%M:%S')

    @staticmethod
    def init_log(init_str):
        """Generate a string to identify a new model run.

        Parameters
        ----------
        init_str : `str`
            The string to write to the model log file.

        """
        LOGGER.info(80 * '-')
        LOGGER.info('{}: '.format(LogConfig.now()) + init_str)
        LOGGER.info(80 * '-')
        
def img2np(path, tile_size=None, tile=None, pad=False, cval=0):
    r"""Read an image to a :py:class:`numpy.ndarray`.

    If ``tile_size`` is not `None`, the input image is divided into square
    tiles of size ``(tile_size, tile_size)``. If the image is not evenly
    divisible and ``pad=False``, a ``ValueError`` is raised. However, if
    ``pad=True``, center padding with constant value ``cval`` is applied.

    The tiling works as follows:

        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |  tile_00  |  tile_01  |    ...    |  tile_0n  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |  tile_10  |  tile_11  |    ...    |  tile_1n  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |    ...    |    ...    |    ...    |    ...    |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+
        |           |           |           |           |
        |  tile_m0  |  tile_m1  |    ...    |  tile_mn  |
        |           |           |           |           |
        +-----------+-----------+-----------+-----------+

    where :math:`m = n`. Each tile has its id, which starts at `0` in the
    topleft corner of the input image, i.e. `tile_00` has :math:`id=0`, and
    increases along the width axis, i.e. `tile_0n` has :math:`id=n`, `tile_10`
    has :math:`id=n+1`, ..., `tile_mn` has :math:`id=(m \\cdot n) - 1`.

    If ``tile`` is an integer, only the tile with ``id=tile`` is returned.

    Parameters
    ----------
    path : `str` or :py:class:`pathlib.Path` or :py:class:`numpy.ndarray`
        The image to read.
    tile_size : `None` or `int`, optional
        The size of a tile. The default is `None`.
    tile : `int`, optional
        The tile id. The default is `None`.
    pad : `bool`, optional
        Whether to center pad the input image. The default is `False`.
    cval : `float`, optional
        The constant padding value. The default is `0`.

    Raises
    ------
    FileNotFoundError
        Raised if ``path`` is a path that does not exist.
    TypeError
        Raised if ``path`` is not `str` or `None` or :py:class:`numpy.ndarray`.

    Returns
    -------
    image : :py:class:`numpy.ndarray`
        The image array. The output shape is:
            - `(tiles, bands, tile_size, tile_size)` if ``tile_size`` is not
            `None`. If the image does only have one band,
            `(tiles, tile_size, tile_size)`

            - `(bands, height, width)` if ``tile_size=None``. If the image does
            only have one band, `(height, width)`.

    """
    # check the type of path
    if isinstance(path, str) or isinstance(path, pathlib.Path):

        # check if the path is a url
        if str(path).startswith('http'):
            # gdal virtual file system for url paths
            img = gdal.Open('/vsicurl/{}'.format(str(path)))
        else:
            # image is stored in a file system
            img = gdal.Open(str(path))

        # number of bands
        bands = img.RasterCount

        # spatial size
        height = img.RasterYSize
        width = img.RasterXSize

        # data type
        dtype = getattr(Gdal2Numpy,
                        gdal.GetDataTypeName(img.GetRasterBand(1).DataType))
        dtype = dtype.value

    elif path is None:
        LOGGER.warning('Path is of NoneType, returning.')
        return

    # accept numpy arrays as input
    elif isinstance(path, np.ndarray):
        # input array
        img = path

        # check the dimensions of the input array
        if img.ndim > 2:
            bands = img.shape[0]
            height = img.shape[1]
            width = img.shape[2]
        else:
            bands = 1
            height = img.shape[0]
            width = img.shape[1]

            # expand input array to fit band dimension
            img = np.expand_dims(img, axis=0)

        # input array data type
        dtype = img.dtype

    else:
        raise TypeError('Input of type {} not supported'.format(type(path)))

    # check whether to read the image in tiles
    if tile_size is None:

        # number of tiles
        ntiles = 1

        # create empty numpy array to store whole image
        image = np.empty(shape=(ntiles, bands, height, width), dtype=dtype)

        # iterate over the bands of the image
        for b in range(bands):

            # read the data of band b
            if isinstance(img, np.ndarray):
                data = img[b, ...]
            else:
                band = img.GetRasterBand(b+1)
                data = band.ReadAsArray()

            # append band b to numpy image array
            image[0, b, :, :] = data

    else:

        # check whether the image is evenly divisible in square tiles
        # of size (tile_size x tile_size)
        ntiles, padding = is_divisible((height, width), tile_size, pad)

        # image size after padding
        y_size = height + padding[0] + padding[2]
        x_size = width + padding[1] + padding[3]

        # print progress
        LOGGER.debug('Image size: {}'.format((height, width)))
        LOGGER.debug('Dividing image into {} tiles of size {} ...'
                     .format(ntiles, (tile_size, tile_size)))
        LOGGER.debug('Padding image (b, l, t, r): {}'.format(tuple(padding)))
        LOGGER.debug('Padded image size: {}'.format((y_size, x_size)))

        # get the indices of the top left corner for each tile
        topleft = tile_topleft_corner((y_size, x_size), tile_size)

        # whether to read all tiles or a single tile
        if tile is not None:
            ntiles = 1

        # create empty numpy array to store the tiles
        image = np.ones((ntiles, bands, tile_size, tile_size),
                        dtype=dtype) * cval

        # iterate over the topleft corners of the tiles
        for k, corner in topleft.items():

            # in case a single tile is required, skip the rest of the tiles
            if tile is not None:
                if k != tile:
                    continue

                # set the key to 0 for correct array indexing when reading
                # a single tile from the image
                LOGGER.debug('Processing tile {} ...'.format(k))
                k = 0
            else:
                LOGGER.debug('Creating tile {} with top-left corner {} ...'
                             .format(k, corner))

            # calculate shift between padded and original image
            row = corner[0] - padding[2] if corner[0] > 0 else corner[0]
            col = corner[1] - padding[1] if corner[1] > 0 else corner[1]
            y_tl = row + padding[2] if row == 0 else 0
            x_tl = col + padding[1] if col == 0 else 0

            # iterate over the bands of the image
            for b in range(bands):

                # check if the current tile extend exists in the image
                nrows, ncols = check_tile_extend(
                    (height, width), (row, col), tile_size)

                # read the current tile from band b
                if isinstance(img, np.ndarray):
                    data = img[b, row:row+nrows, col:col+ncols]
                else:
                    band = img.GetRasterBand(b+1)
                    data = band.ReadAsArray(col, row, ncols, nrows)

                # append band b to numpy image array
                image[k, b, y_tl:nrows, x_tl:ncols] = data[0:(nrows - y_tl),
                                                           0:(ncols - x_tl)]

    # check if there are more than 1 band
    if not bands > 1:
        image = image.squeeze(axis=1)

    # check if there are more than 1 tile
    if not ntiles > 1:
        image = image.squeeze(axis=0)

    # close tif file
    del img

    # return the image
    return image
