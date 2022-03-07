from neurotorch.nets.RSUNetMulti import RSUNetMulti
from neurotorch.core.predictor_multilabel import Predictor
from neurotorch.datasets.filetypes import TiffVolume
from neurotorch.datasets.dataset import Array
from neurotorch.datasets.datatypes import (BoundingBox, Vector)
from multiprocessing import Pool
import numpy as np
import tifffile as tif
import os
import glob
import argparse
import natsort
import pandas as pd
from datetime import date
import torch

def predict(checkpoint, specimen_dir, chunk_dir, bb, ids, error_list, gpu, files_per_chunk):

    # Step 1. Make Segmentation Output DIrectory 
    try:
        chunk_dir_base_path = os.path.basename(chunk_dir)
        
        if 'Left' in chunk_dir_base_path:
            seg_dir = os.path.join(specimen_dir,'Left_Segmentation')
        elif 'Right' in chunk_dir_base_path:
            seg_dir = os.path.join(specimen_dir,'Right_Segmentation')
        else:
            seg_dir = os.path.join(specimen_dir,'Segmentation')
            
        if not os.path.isdir(seg_dir):
            os.mkdir(seg_dir)

    except:
        print('erro making segmentation directory')
        error_list.append(str(ids)+ ' -making directory')
   
    # Step 2. Run segmentation 
    try:
        net = RSUNetMulti()
        count = [0,0,0]
        number_of_small_segments = len([ff for ff in os.listdir(chunk_dir) if '.tif' in ff])
        print('I think there are {} chunk tiff files in {}'.format(number_of_small_segments,chunk_dir))
        for n in range(1,number_of_small_segments+1):
            bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], files_per_chunk))

            nth_tiff_stack = os.path.join(chunk_dir,'chunk{}.tif'.format(n))
            with TiffVolume(nth_tiff_stack, bbn) as inputs:                         
                
                # Predict
                predictor = Predictor(net, checkpoint, gpu_device=gpu)
                # Output_volume is a list (len3) of Arrays for each of 3 foreground channels (soma, axon, dendrite)
                output_volume = [Array(np.zeros(inputs.getBoundingBox().getNumpyDim(), dtype=np.uint8)) for _ in range(3)] 
                print('bb0', inputs.getBoundingBox())
                predictor.run(inputs, output_volume)      
                
                for ch in range(3):
                    ch_dir = os.path.join(seg_dir,'ch%d'%(ch+1))
                    if not os.path.isdir(ch_dir):
                        os.mkdir(ch_dir)
                    probability_map = output_volume[ch].getArray()
                    for i in range(probability_map.shape[0]): # Save as multiple tif files
                        #print('Prob Map Shape= ', probability_map.shape[0])
                        count[ch] +=1
                        tif.imsave(os.path.join(ch_dir,'%03d.tif'%(count[ch])), probability_map[i,:,:])                             
    except:
        print('error with segmentation')
        error_list.append(str(ids)+ ' -segmentation')  
    
     
    # Step 3. Remove Duplicate Files if necessary 
    try:
        
        individual_tif_dir = os.path.join(specimen_dir,'Single_Tif_Images')
        number_of_individual_tiffs = len([f for f in os.listdir(individual_tif_dir) if '.tif' in f])
        for ch in range(3):
            ch_dir = os.path.join(seg_dir,'ch%d'%(ch+1))
            number_of_segmented_tiffs =  len([f for f in os.listdir(ch_dir) if '.tif' in f])
            print('Number of individual tiffs = {}'.format(number_of_individual_tiffs))
            print('Number of segmented tiffs = {}'.format(number_of_segmented_tiffs))

            number_of_duplicates = number_of_segmented_tiffs-number_of_individual_tiffs
                    # assigning the number of duplicates to the difference in length between segmented dir and individual tiff dir 
            if number_of_duplicates == 0: 
                print('no duplicates were made')
                print('num duplicates = {}'.format(number_of_duplicates))

            else:
                print('num duplicates = {}'.format(number_of_duplicates))
                # this means that list_of_segmented_files[-files_per_chunk:-number_of_suplicates] can be erased because of part 7 in preprocessing
                list_of_segmented_files = [x for x in natsort.natsorted(os.listdir(ch_dir)) if '.tif' in x]
                second_index = files_per_chunk-number_of_duplicates
                duplicate_segmentations = list_of_segmented_files[-files_per_chunk:-(second_index)] 
                print(duplicate_segmentations)       

                for files in duplicate_segmentations:
                    os.remove(os.path.join(ch_dir,files))
    except:
        print('error with removing files')
        error_list.append(str(ids)+' -removing duplicates')

    return error_list


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', '-c', type=str, help='path to checkpoint')
    parser.add_argument('--outdir', '-v', type=str, help='directory of validation/test data')
    parser.add_argument('--csv', type=str, help='input csv with specimen id column header')
    parser.add_argument('--num_processes', type=int, default = 1, help='number of processes to run parallel')
    parser.add_argument('--gpu', type=int, default = 0, help='gpu device')
    parser.add_argument('--chunk', type=int, default = 32, help='files per chunk')
    
    today = date.today()
    todays_date = today.strftime("%b_%d_%Y")

    args = parser.parse_args()
    outdir = args.outdir
    df = pd.read_csv(args.csv)
    specimens = list(df.specimen_id.values)
    files_per_chunk = args.chunk
    num_processes = args.num_processes
    
    all_error_list = []
    if num_processes == 1:

        for sp_ids in specimens:       
            specimen_dir = os.path.join(outdir,str(sp_ids))
            error_list = []
            error_list_2 = []

            if os.path.exists(os.path.join(specimen_dir,'Chunks_of_%d_Left'%files_per_chunk)):
                # chunk_dirs 
                chunk_dir_left = os.path.join(specimen_dir,'Chunks_of_%d_Left'%files_per_chunk)
                chunk_dir_right = os.path.join(specimen_dir,'Chunks_of_%d_Right'%files_per_chunk)

                # bboxes
                left_bbox_path = os.path.join(outdir,str(sp_ids),'bbox_{}_Left.csv'.format(sp_ids))
                right_bbox_path = os.path.join(outdir,str(sp_ids),'bbox_{}_Right.csv'.format(sp_ids))
                df_l = pd.read_csv(left_bbox_path)
                df_r = pd.read_csv(right_bbox_path)
                bb_l = df_l.bound_boxing.values
                bb_r = df_r.bound_boxing.values

                # Predict
                res_l = predict(args.ckpt, specimen_dir, chunk_dir_left, bb_l, sp_ids, error_list, args.gpu, files_per_chunk)
                res_r = predict(args.ckpt, specimen_dir, chunk_dir_right, bb_r, sp_ids, error_list_2, args.gpu, files_per_chunk)

                all_error_list.append(res_l)
                all_error_list.append(res_r)
            
            else:
                # chunk dir
                chunk_dir = os.path.join(specimen_dir,'Chunks_of_%d'%files_per_chunk)
                
                # bboxes
                bbox_path = os.path.join(outdir,str(sp_ids),'bbox_{}.csv'.format(sp_ids))
                df = pd.read_csv(bbox_path)
                bb = df.bound_boxing.values

                # predict
                res = predict(args.ckpt, specimen_dir, chunk_dir, bb, sp_ids, error_list, args.gpu, files_per_chunk)
                all_error_list.append(res)


    else:
        p = Pool(processes=num_processes)
        parallel_input = [(args.ckpt, int(i), pd.read_csv(os.path.join(outdir,str(i),'bbox_{}.csv'.format(sp_ids))).bound_boxing.values, error_list) for i in specimens]
        all_error_list = p.starmap(predict, parallel_input)

    with open('{}_gpu{}_segmentation_error_log.txt'.format(todays_date, str(args.gpu)), 'w') as f:
        for item in all_error_list:
            f.write("%s\n" % item)
