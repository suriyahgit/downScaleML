# Let's load the packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import joblib

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})

model = joblib.load("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/random_forest_exp/Models/tasmean/RandomForest_tasmean_ztuvq_500_850_mslp_dem_sa_doy_1985_to_1994_3_10.joblib")
f = model.feature_importances_
print(f)
