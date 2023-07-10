#!/bin/bash

# activate conda environment
conda activate climax

# move to project repository
cd /home/sdhinakaran/eurac/downScaleML/util_modules/

python cli_download_CERRA-Copy1.py /mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/CERRA/pr/ --variable total_precipitation -sy 1995 -ey 2004



