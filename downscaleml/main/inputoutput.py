import pathlib

ERA5="REANALYSIS"
#ERA5="p_REANALYSIS"

ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/hydroModelDownscale/')

# path to this file
HERE = pathlib.Path(__file__).parent

ERA5_PATH = ROOT.joinpath(ERA5)

SEAS5_PATH = ROOT

OBS_PATH = ROOT.joinpath('CERRA')

DEM_PATH = ROOT.joinpath('DEM')

RESULTS = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/base_models/')

MODEL_PATH = RESULTS.joinpath('RESULTS_model')

TARGET_PATH = RESULTS.joinpath('RESULTS')