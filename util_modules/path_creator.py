import os
import shutil

# specify the source and destination paths
src_path = "/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/ERA5/Hourly"
dst_path = "/mnt/CEPH_PROJECTS/InterTwin/02_Original_Climate_Data/ERA5/Daily"

# use os.walk() to iterate through all directories and subdirectories in the source path
for root, dirs, files in os.walk(src_path):

    # create the corresponding directory structure in the destination path
    for directory in dirs:
        src_dir = os.path.join(root, directory)
        dst_dir = src_dir.replace(src_path, dst_path)
        os.makedirs(dst_dir, exist_ok=True)

