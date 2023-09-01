import sys
import xarray as xr
import logging
from logging.config import dictConfig

from downscaleml.core.utils import reproject_cdo, LogConfig
from downscaleml.core.cli import dem_preprocess_parser

from downscaleml.core.logging import log_conf

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    # initialize logging
    dictConfig(log_conf())

    # define command line argument parser
    parser = dem_preprocess_parser()

    # parse command line arguments
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        sys.exit()
    else:
        args = parser.parse_args(args)

    # check whether the source directory exists
    if args.source.exists() and args.target:
        
        source = args.source
        target = args.target
        
        # reproject and resample to target grid in parallel
        if args.reproject:
            LOGGER.info('Reprojecting and resampling to target grid ...')

            # check whether the target grid file exists
            if not args.grid.exists():
                LOGGER.info('{} does not exist.'.format(args.grid))
                sys.exit()

            reprojected_path = reproject_cdo(args.grid, source, target, args.mode, args.overwrite)
            ds = xr.open_dataset(reprojected_path)
            ds = ds.rename({'longitude': 'x','latitude': 'y'})
            reprojected_path.unlink()
            ds.to_netcdf("{}/{}".format(target.parent, target.name))
            LOGGER.info('DEM {} Processed and saved in target_location'.format(target.name))
    else:
        LOGGER.info('Provide proper source and target please...')
        sys.exit()