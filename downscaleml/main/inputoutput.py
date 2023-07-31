import pathlib

NET = "RandomForest"

ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/larger_alps/')

# path to this file
HERE = pathlib.Path(__file__).parent

ERA5_PATH = ROOT.joinpath('REANALYSIS')

OBS_PATH = ROOT.joinpath('CERRA')

DEM_PATH = ROOT.joinpath('DEM')

RESULTS = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/larger_alps/')

MODEL_PATH = RESULTS.joinpath('ml_Models')

TARGET_PATH = RESULTS.joinpath('ml_Predictions')