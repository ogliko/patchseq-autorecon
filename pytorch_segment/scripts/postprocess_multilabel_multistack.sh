#!/bin/bash
#SBATCH --partition=celltypes
#SBATCH --job-name=post_run0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240g
#SBATCH --time=10:00:00
#SBATCH --output=/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/segmentation/neurotorch/sbatch_outputs/0_0.out
cd /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/segmentation/neurotorch/
source activate neurotorch
python postprocess_multilabel_multistack.py --outdir test_dir --csv csv_inputs/00_00.csv
