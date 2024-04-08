#!/usr/bin/env bash

# activate conda environment
micromamba activate check

# move to project repository
cd /home/sdhinakaran/eurac/downScaleML/

# predictands
#PREDICTAND=(tasmin tasmax tasmean pr)
PREDICTAND=(tasmean pr)

# wet day thresholds to test
#WET_DAY_THRESHOLDS=(0 0.5 1 2 3 5)
WET_DAY_THRESHOLDS=(0)

#MODELS=(RandomForestRegressor XGBRegressor AdaBoostRegressor LGBMRegressor)
MODELS=(LGBMRegressor RandomForestRegressor XGBRegressor AdaBoostRegressor)

# iterate over predictands
for predictand in ${PREDICTAND[@]}; do

    # change predictand in configuration
    sed -i "s/PREDICTAND\s*=.*/PREDICTAND='$predictand'/" ./downscaleml/main/config.py
    echo "Predictand = $predictand"

    # SGD with fixed and cyclic learning rate policy
    if [ "$predictand" = "pr" ]; then
        sed -i "s/STRATIFY\s*=.*/STRATIFY = True/" ./downscaleml/main/config.py
        
        for wet in ${WET_DAY_THRESHOLDS[@]}; do
            # change learning rate scheduler in configuration
            sed -i "s/WET_DAY_THRESHOLD\s*=.*/WET_DAY_THRESHOLD=$wet/" ./downscaleml/main/config.py
            echo "WET = $wet"
            
            # iterate over weight decay values
            for model in ${MODELS[@]}; do
                # change weight regularization in configuration
                sed -i "s/NET\s*=.*/NET='$model'/" ./downscaleml/main/config.py
                echo "Model = $model"

                # run downscaling
                python downscaleml/main/t_program_ml_downscale_check_doy.py
            done
        done
    else
        sed -i "s/STRATIFY\s*=.*/STRATIFY = False/" ./downscaleml/main/config.py
        
        # iterate over weight decay values
        for model in ${MODELS[@]}; do
            # change weight regularization in configuration
            sed -i "s/NET\s*=.*/NET='$model'/" ./downscaleml/main/config.py
            echo "Model = $model"

            # run downscaling
            python downscaleml/main/t_program_ml_downscale_check_doy.py
        done
    fi
done
        
        