"""Command line argument parsers."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-

# builtins
import argparse
import logging
import pathlib

# locals
from climax.core.constants import (CDO_RESAMPLING_MODES, ERA5_VARIABLES)

# epilogue to display at the end of each parser
EPILOGUE = 'Author: Suriyah Dhinakaran, pythonsuriyah@gmail.com'

# module level logger
LOGGER = logging.getLogger(__name__)

# parser to preprocess ERA5 data: climax.main.preprocess.preprocess_ERA5.py
def preprocess_era5_parser():

    # define command line argument parser
    parser = argparse.ArgumentParser(
        description='Aggregate ERA5 reanalysis from hourly to daily data.',
        epilog=EPILOGUE,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=50, indent_increment=2))

    # positional arguments

    # positional argument: path to search for ERA5 NetCDF files
    parser.add_argument('source', type=pathlib.Path,
                        help='Path to search for ERA5 NetCDF files.')

    # positional argument: path to save the reprojected NetCDF files
    parser.add_argument('target', type=pathlib.Path,
                        help='Path to save the aggregated ERA5 NetCDF files.')

    # optional arguments

    # default values
    default = '(default: %(default)s)'


    # optional argument: name of the variable of interest
    parser.add_argument('-v', '--variable', type=str,
                        help='Name of the variable of interest.',
                        choices=ERA5_VARIABLES, default=None, nargs='+',
                        metavar='')

    # optional argument: whether to reproject to target grid
    parser.add_argument('-r', '--reproject', type=bool,
                        help=('Whether to reproject to target grid {}.'
                              .format(default)), default=False, nargs='?',
                        const=True, metavar='')

    # optional argument: path to the target grid file
    parser.add_argument('-g', '--grid', type=pathlib.Path,
                        help='Path to the target grid {}.'.format(default),
                        default=None, metavar='')

    # optional argument: whether to overwrite files
    parser.add_argument('-o', '--overwrite', type=bool,
                        help='Overwrite existing files {}.'.format(default),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: whether to apply compression
    parser.add_argument('-c', '--compress', type=bool,
                        help='Apply file compression {}.'.format(default),
                        default=False, nargs='?', const=True, metavar='')

    # optional argument: dry run, print files which would be processed
    parser.add_argument('-d', '--dry-run', type=bool,
                        help=('Print files which would be processed {}.'
                              .format(default)), default=False, nargs='?',
                        const=True, metavar='')

    # optional argument: resampling mode
    parser.add_argument('-m', '--mode', type=str,
                        help='Resampling mode {}.'.format(default),
                        default='bilinear', choices=CDO_RESAMPLING_MODES,
                        metavar='')

    return parser
