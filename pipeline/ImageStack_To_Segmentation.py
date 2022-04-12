import numpy as np
from neurotorch.nets.RSUNetMulti import RSUNetMulti
from neurotorch.core.predictor_multilabel import Predictor
from neurotorch.datasets.filetypes import TiffVolume
from neurotorch.datasets.dataset import Array
from neurotorch.datasets.datatypes import (BoundingBox, Vector)
import tifffile as tif
import os
import natsort
import pandas as pd
from datetime import date
import argschema as ags

class InputSchema(ags.ArgSchema):
    ckpt = ags.fields.InputFile(description='checkpoint file to use for segmentation ')
    specimen_dir = ags.fields.InputDir(description="specimen directory")
    specimen_id = ags.fields.Str(default=None,description="specimen id")
    gpu = ags.fields.Int(default=0, description = "gpu to use")
    raw_single_tif_dir = ags.fields.InputDir(description='raw image directory')

def validate(checkpoint, specimen_dir, chunk_dir, raw_single_tif_dir,  bb, ids, error_list, gpu):
    """
    Will run run segmentation on input directory of 3d tif volumes. These volumes should have dimensions:
    64*n x 64*m x 32. Segmentation will create a subdirectory in the specimen_dir named Segmentation
    This directory will contain 3 output channels. Ch1 = soma, Ch2 = axon, Ch3 = dendrite

    :param checkpoint: checkpoint file
    :param specimen_dir: specimens directory
    :param chunk_dir: directory within specimens directory that has tif volumes
    :param raw_single_tif_dir: original single tif image directory
    :param bb: bounding box specifying segmentation dimensions, created in PreProcessing_ImageStack.py
    :param ids: specimen id
    :param error_list: empty list populated with any errors that occur
    :param gpu: 0
    :return: None
    """
    files_per_chunk = 32
    print("Using checkpoint file: {}".format(checkpoint))
    print("Going to load checkpoint file, does it exist? {}".format(os.path.exists(checkpoint)))

    #Step 1. Make Segmentation Output DIrectory 
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
        print('error making segmentation directory')
        error_list.append(str(ids)+ ' -making directory')
   
    #Step 2. Run segmentation 
    if True:#try:
        net = RSUNetMulti()
        predictor = Predictor(net, checkpoint, gpu_device=gpu)

        count = [0,0,0]
        number_of_small_segments = len([ff for ff in os.listdir(chunk_dir) if '.tif' in ff])
        print('I think there are {} chunk tiff files in {}'.format(number_of_small_segments,chunk_dir))
        for n in range(1,number_of_small_segments+1):
            bbn = BoundingBox(Vector(bb[0], bb[1], bb[2]), Vector(bb[3], bb[4], files_per_chunk))

            nth_tiff_stack = os.path.join(chunk_dir,'chunk{}.tif'.format(n))
            with TiffVolume(nth_tiff_stack, bbn) as inputs:                         
                
                # Predict
                predictor = Predictor(net, checkpoint, gpu_device=gpu)
                # output_volume is a list (len3) of Arrays for each of 3 foreground channels (soma, axon, dendrite)
                output_volume = [Array(np.zeros(inputs.getBoundingBox().getNumpyDim(), dtype=np.uint8)) for _ in range(3)] 
                print('bb0', inputs.getBoundingBox())
                predictor.run(inputs, output_volume)      
                
                for ch in range(3):
                    ch_dir = os.path.join(seg_dir,'ch%d'%(ch+1))
                    if not os.path.isdir(ch_dir):
                        os.mkdir(ch_dir)
                    probability_map = output_volume[ch].getArray()
                    for i in range(probability_map.shape[0]): # save as multiple tif files
                        #print('Prob Map Shape= ', probability_map.shape[0])
                        count[ch] +=1
                        tif.imsave(os.path.join(ch_dir,'%03d.tif'%(count[ch])), probability_map[i,:,:])                             
    else:#except:
        print('error with segmentation')
        error_list.append(str(ids)+ ' -segmentation')  
    
     
    #Step 3. Remove Duplicate Files if necessary 
    try:
        
        number_of_individual_tiffs = len([f for f in os.listdir(raw_single_tif_dir) if '.tif' in f])
        for ch in range(3):
            ch_dir = os.path.join(seg_dir,'ch%d'%(ch+1))
            number_of_segmented_tiffs =  len([f for f in os.listdir(ch_dir) if '.tif' in f])
            print('Number of individual tiffs = {}'.format(number_of_individual_tiffs))
            print('Number of segmented tiffs = {}'.format(number_of_segmented_tiffs))

            number_of_duplicates = number_of_segmented_tiffs-number_of_individual_tiffs
                    #assigning the number of duplicates to the difference in length between segmented dir and individual tiff dir. 
            if number_of_duplicates == 0: 
                print('no duplicates were made')
                print('num duplicates = {}'.format(number_of_duplicates))

            else:
                print('num duplicates = {}'.format(number_of_duplicates))
                #this means that list_of_segmented_files[-32:-number_of_suplicates] can be erased because of part 7 in preprocessing
                list_of_segmented_files = [x for x in natsort.natsorted(os.listdir(ch_dir)) if '.tif' in x]
                second_index = 32-number_of_duplicates
                duplicate_segmentations = list_of_segmented_files[-32:-(second_index)] 
                print(duplicate_segmentations)       

                for files in duplicate_segmentations:
                    os.remove(os.path.join(ch_dir,files))
    except:
        print('error with removing files')
        error_list.append(str(ids)+' -removing duplicates')

    return error_list


def main(ckpt, specimen_dir, raw_single_tif_dir, specimen_id, gpu, **kwargs ):

    today = date.today()
    todays_date = today.strftime("%b_%d_%Y")

    chunk_dir_left = os.path.join(specimen_dir, 'Chunks_of_32_Left')
    if os.path.exists(chunk_dir_left):
        #chunk_dirs
        chunk_dir_right = os.path.join(specimen_dir,'Chunks_of_32_Right')

        #bboxes
        left_bbox_path = os.path.join(specimen_dir,'bbox_{}_Left.csv'.format(specimen_id))
        right_bbox_path = os.path.join(specimen_dir,'bbox_{}_Right.csv'.format(specimen_id))
        df_l = pd.read_csv(left_bbox_path)
        df_r = pd.read_csv(right_bbox_path)
        bb_l = df_l.bound_boxing.values
        bb_r = df_r.bound_boxing.values

        #Validate
        validate(ckpt, specimen_dir, chunk_dir_left, raw_single_tif_dir, bb_l, specimen_id, [], gpu)
        validate(ckpt, specimen_dir, chunk_dir_right, raw_single_tif_dir, bb_r, specimen_id, [], gpu)


    else:
        #chunk dir
        chunk_dir = os.path.join(specimen_dir,'Chunks_of_32')

        #bboxes
        bbox_path = os.path.join(specimen_dir,'bbox_{}.csv'.format(specimen_id))
        df = pd.read_csv(bbox_path)
        bb = df.bound_boxing.values

        #validate
        validate(ckpt, specimen_dir, chunk_dir, raw_single_tif_dir, bb, specimen_id, [], gpu)

if __name__ == "__main__":
	module = ags.ArgSchemaParser(schema_type=InputSchema)
	main(**module.args)
