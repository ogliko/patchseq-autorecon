#!bin/bash

raw_single_tif_dir_tarball="Example_Specimen_2112/Example_Input_Stack.tar.gz"
specimen_dir="Example_Specimen_2112/"
specimen_id=2112
raw_single_tif_dir="Example_Specimen_2112/Example_Input_Stack/"
invert_image_color="True"
ckpt="aspiny_model.ckpt"
intensity_threshold=252

tar -xzvf ${raw_single_tif_dir_tarball} -C ${specimen_dir}

echo "RUNNING IMAGE STACK PRE PROCESSING"
python PreProcess_ImageStack.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \
--raw_single_tif_dir ${raw_single_tif_dir}

echo "RUNNING SEGMENTATION"
python ImageStack_To_Segmentation.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \
--raw_single_tif_dir ${raw_single_tif_dir} \
--ckpt ${ckpt}

echo "RUNNING SEGMENTATION TO SKELETON"
python Segmentation_To_Skeleton.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \
--intensity_threshold ${intensity_threshold}

echo "RUNNING SKELETON TO SWC"
python Skeleton_To_Swc.py --specimen_dir ${specimen_dir} \
--specimen_id ${specimen_id} \

echo "Example Pieline Completed Succesfully "