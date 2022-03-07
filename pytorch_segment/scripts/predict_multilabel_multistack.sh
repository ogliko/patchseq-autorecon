#!/bin/bash
#SBATCH --partition=celltypes
#SBATCH --job-name=seg_run0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=v100:1
#SBATCH --mem=62g
#SBATCH --time=10:00:00
#SBATCH --output=/allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/segmentation/neurotorch/sbatch_outputs/0_0.out
cd /allen/programs/celltypes/workgroups/mousecelltypes/MachineLearning/Olga/segmentation/neurotorch/
source activate neurotorch
python predict_multilabel_multistack.py --ckpt model1.ckpt --outdir test_dir --csv csv_inputs/0.csv --gpu 0 --chunk 32
