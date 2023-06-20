import pathlib

NET = "RandomForest"

ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/data/')

# path to this file
HERE = pathlib.Path(__file__).parent

ERA5_PATH = ROOT.joinpath('REANALYSIS')

OBS_PATH = ROOT.joinpath('CERRA')

DEM_PATH = ROOT.joinpath('DEM')

RESULTS = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/glm_skill/')

MODEL_PATH = RESULTS.joinpath('Models')

TARGET_PATH = RESULTS.joinpath('Predictions')