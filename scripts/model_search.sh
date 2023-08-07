#!/usr/bin/env bash

# activate conda environment
conda activate climax

# move to project repository
cd /home/sdhinakaran/eurac/downScaleML/

# predictands
PREDICTAND=(tasmin tasmax tasmean pr)

# wet day thresholds to test
WET_DAY_THRESHOLDS=(0 0.5 1 2 3 5)

MODELS=(RandomForestRegressor XGBRegressor AdaBoostRegressor LGBMRegressor)

TAS='["mean_sea_level_pressure", "2m_temperature"]'

PR='["mean_sea_level_pressure", "total_precipitation"]'

# iterate over predictands
for predictand in ${PREDICTAND[@]}; do

    # change predictand in configuration
    sed -i "s/PREDICTAND\s*=.*/PREDICTAND='$predictand'/" ./downscaleml/main/config.py

    # SGD with fixed and cyclic learning rate policy
    if [ "$predictand" = "pr" ]; then
        sed -i "s/ERA5_S_PREDICTORS\s*=.*/ERA5_S_PREDICTORS=$PR/" ./downscaleml/main/config.py
        sed -i "s/STRATIFY\s*=.*/STRATIFY = True/" ./downscaleml/main/config.py
        
        for wet in ${WET_DAY_THRESHOLDS[@]}; do
            # change learning rate scheduler in configuration
            sed -i "s/WET_DAY_THRESHOLD\s*=.*/WET_DAY_THRESHOLD=$wet/" ./downscaleml/main/config.py
            
            # iterate over weight decay values
            for model in ${MODELS[@]}; do
                # change weight regularization in configuration
                sed -i "s/NET\s*=.*/NET='$model'/" ./downscaleml/main/config.py

                # run downscaling
                python downscaleml/main/program_ml_downscale.py
            done
        done
    else
        sed -i "s/ERA5_S_PREDICTORS\s*=.*/ERA5_S_PREDICTORS=$TAS/" ./downscaleml/main/config.py
        sed -i "s/STRATIFY\s*=.*/STRATIFY = False/" ./downscaleml/main/config.py
        
        # iterate over weight decay values
        for model in ${MODELS[@]}; do
            # change weight regularization in configuration
            sed -i "s/NET\s*=.*/NET='$model'/" ./downscaleml/main/config.py

            # run downscaling
            python downscaleml/main/program_ml_downscale.py
        done
    fi
done
        
        