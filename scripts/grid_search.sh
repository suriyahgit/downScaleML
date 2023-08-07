#!/usr/bin/env bash

# activate conda environment
conda activate climax

# move to project repository
cd /home/sdhinakaran/eurac/downScaleML/

# predictands
#PREDICTAND=(tasmin tasmax tasmean)
PREDICTAND=(pr)

# optimizers
# OPTIM=(torch.optim.Adam torch.optim.SGD)
OPTIM=(torch.optim.Adam)

# learning rate scheduler
LRSCHEDULER=(None torch.optim.lr_scheduler.CyclicLR)

# wet day thresholds to test
WET_DAY_THRESHOLDS=(0 0.5 1 2 3 5)

# weight decay values to test
LAMBDA=(0 0.000001 0.00001 0.0001 0.001 0.01 1)

# change flag for sensitivity analysis in configuration
sed -i "s/SENSITIVITY\s*=.*/SENSITIVITY\=True/" ./downscaleml/main/config.py

# iterate over predictands
for predictand in ${PREDICTAND[@]}; do

    # change predictand in configuration
    sed -i "s/PREDICTAND\s*=.*/PREDICTAND='$predictand'/" ./downscaleml/main/config.py

    # define available loss functions for current predictand
    if [ "$predictand" = "pr" ]; then
        LOSS=(L1Loss BernoulliGammaLoss MSELoss)
    else
        LOSS=(L1Loss MSELoss)
    fi

    # iterate over loss functions
    for loss in ${LOSS[@]}; do

        # change loss function in configuration
        if [ "$loss" = "L1Loss" ] || [ "$loss" = "MSELoss" ]; then
            sed -i "s/LOSS\s*=.*/LOSS=$loss()/" ./downscaleml/main/config.py
        else
    	    sed -i "s/LOSS\s*=.*/LOSS=$loss(min_amount=1)/" ./downscaleml/main/config.py
        fi

        # iterate over the optimizer
        for optim in ${OPTIM[@]}; do

            # change optimizer in configuration
            sed -i "s/OPTIM\s*=.*/OPTIM=$optim/" ./downscaleml/main/config.py

            # SGD with fixed and cyclic learning rate policy
            if [ "$optim" = "torch.optim.SGD" ]; then
                for scheduler in ${LRSCHEDULER[@]}; do
                    # change learning rate scheduler in configuration
                    sed -i "s/LR_SCHEDULER\s*=.*/LR_SCHEDULER=$scheduler/" ./downscaleml/main/config.py

                    # iterate over weight decay values
                    for lambda in ${LAMBDA[@]}; do
                        # change weight regularization in configuration
                      	sed -i "s/'weight_decay':.*/'weight_decay': $lambda/" ./downscaleml/main/config.py

                       	# run downscaling
                        python downscaleml/main/program_ml_downscale.py
                    done
                done
            else
                # do not use a LR-scheduler with Adam
                sed -i "s/LR_SCHEDULER\s*=.*/LR_SCHEDULER=None/" ./downscaleml/main/config.py

                # iterate over weight decay values
                for lambda in ${LAMBDA[@]}; do
                    # change weight regularization in configuration
                    sed -i "s/'weight_decay':.*/'weight_decay': $lambda/" ./downscaleml/main/config.py

                    # run downscaling
                    python climax/main/downscale_master.py
                done
            fi
        done
    done
done
