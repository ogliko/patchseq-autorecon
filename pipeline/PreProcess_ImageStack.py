import os
import pandas as pd
import cv2
import shutil
import argschema as ags
import natsort
import numpy as np
from tifffile import imsave, imread
import psutil

class InputSchema(ags.ArgSchema):

    specimen_id = ags.fields.Str(description='specimen id')
    raw_single_tif_dir = ags.fields.InputDir(description="A directory with individual tif files (z-slices)")
    specimen_dir = ags.fields.InputDir(default=None,description="Directory for specimen output files. If none, use basedir of raw_single_tif_dir")
    invert_image_color = ags.fields.Boolean(default=True,description="Neural network will expect inverted (black background) images")

def stack_into_chunks(chunk_size,raw_single_tif_dir,chunk_dir,ids):
    """
    Will stack a directory of single tif images into one 3d volume tif image. Assumes the stacks are named in
    chronological order.

    :param chunk_size: integer. number of tif slices per 3d chunk
    :param raw_single_tif_dir: input directory that has the individual slices of tif images
    :param chunk_dir: where to save the 3d tif files
    :param ids: specimen id
    :return:
    """

    chunk_n = 0
    counter = 0
    cv_stack = []
    list_of_files = [ii for ii in natsort.natsorted(os.listdir(raw_single_tif_dir)) if '.tif' in ii]
    # print('{} Stacking slices into 3D tif chunks'.format(ids))
    for files in list_of_files:
        counter+=1
        # print(files,counter)
        img = cv2.imread(os.path.join(raw_single_tif_dir,files),cv2.IMREAD_UNCHANGED)
        cv_stack.append(img)
        if counter == chunk_size:
            chunk_n+=1
            cv_stack = np.asarray(cv_stack)
            imsave(os.path.join(chunk_dir,'chunk{}.tif'.format(chunk_n)),cv_stack)
            cv_stack = []
            counter = 0
    #if the number of single tif files was a multiple of the chunk_size (usually unlikely)
    if (float(len(list_of_files))/float(chunk_size)).is_integer():
        print('{} files, {} chunk size'.format(len(list_of_files),chunk_size))
        print('The Last Chunk was a multiple of {}'.format(chunk_size))
    #otherwise make one last chunk that has overlap so that we ensure all 3d chunks have cnosistent z-dimension
    else:
        chunk_n+=1
        last_counter = 0
        last_cv_stack = []
        for files in list_of_files[-chunk_size:]:
            last_counter+=1
            last_img = cv2.imread(os.path.join(raw_single_tif_dir,files),cv2.IMREAD_UNCHANGED)
            last_cv_stack.append(last_img)
            if last_counter == chunk_size:
                last_cv_stack = np.asarray(last_cv_stack)
                imsave(os.path.join(chunk_dir,'chunk{}.tif'.format(chunk_n)),last_cv_stack)


def check_for_size_limit(chunk_dir):
    """
    Will check and see if any of the file sizes in the input directory are greater than the available memory.
    If so a warning meessage will be printed to standard output

    :param chunk_dir: Input directory
    :return:
    """
    memory_dict = dict(psutil.virtual_memory()._asdict())
    available_memory = memory_dict['available']

    for tif_stacks in [f for f in os.listdir(chunk_dir) if '.tif' in f]:
        tif_stack_image = os.path.join(chunk_dir,tif_stacks)
        tif_stack_size = os.path.getsize(tif_stack_image)

        if tif_stack_size > available_memory:
            print("WARNING: File {} is {} bytes. Your machine has {} bytes of available memory.".format(tif_stack_image,tif_stack_size,available_memory))
            print("This may lead to crashing")
            # sys.exit()


def myround64(x, base=64):
    return base * int(x/base)

def myround16(x,base=16):
    return base*int(x/base)

def process_specimen(ids,specimen_dir,raw_single_tif_dir,invert_image_color):
    """
    Worker function for script that will do a number of pre-processing steps. Mostly focused on putting the input images
    into a format (dimensions and color inversion) the neural network will be compatible with. The network was trained with
    a patch size of 64x64x32 so we need to get images into nxmx32 dimension where n and m are nearest multiple of 64.

    This script expects an input directory of single tif images (not 3d tif volumes) named in naturally ascending
    order (i.e. 1.tif, 2.tif, 3.tif...) and will run the following:

    -- Get crop dimensions so input images are compatible with neural_network patch size
    -- Crop the images
    -- Stack the slices into chunks of 32 (check for memory limit)
    -- If number of slices is not a multiple of 32, there will be overlap in segmentation that is accounted for
    in ImageStack_To_Segmentation.py
    -- If memory limit is exceeded try splitting images into left and right (TODO update this to dynamically split in
    scenarios where left and right split still exceeds memory)
    -- Create raw input max intensity projections

    :param ids: specimen id
    :param specimen_dir: root directory for specimen
    :param raw_single_tif_dir: input directory of single tif images
    :param invert_image_color: boolean to invert images or not
    :return:
    """

    error_list = []

    #Step 0. Define the chunk size for step 7
    chunk_size = 32

    #Step 1 was removed because it was not useful for consumers outside AIBS

    #Step 2. Choses last file in list of tif files and extracts crop dimensions
    try:
        list_of_files = os.listdir(raw_single_tif_dir)
        for files in list_of_files:
            if files.endswith('.tif'):
                filename_to_extract_crop_info = files

        # print('finding crop dimensions for {}'.format(ids))
        uncropped_img = cv2.imread(os.path.join(raw_single_tif_dir,filename_to_extract_crop_info),cv2.IMREAD_UNCHANGED)

        height, width = uncropped_img.shape

        height_nearest_mult_below = myround64(height)
        width_nearest_mult_below = myround64(width)

        x1,y1 = 0,0
        x2,y2 = width_nearest_mult_below,height_nearest_mult_below

        assert ((y2-y1)/64).is_integer() & ((x2-x1)/64).is_integer()

    except:
        print('{} coordinates for x-y crop are not divisible by 64'.format(ids))
        error_list.append('{} Step 2'.format(ids))


    #Step 3. Crop and invert each image
    try:
        for f in os.listdir(raw_single_tif_dir):
            if f.endswith('.tif'):
                img = cv2.imread(os.path.join(raw_single_tif_dir,f),cv2.IMREAD_UNCHANGED)
                cropped_img = img[y1:y2,x1:x2]

                if invert_image_color:
                    cropped_img_inverted = 255 - cropped_img
                    cv2.imwrite(os.path.join(raw_single_tif_dir,f), cropped_img_inverted)
                else:
                    cv2.imwrite(os.path.join(raw_single_tif_dir, f), cropped_img)

    except:
        print('Unable to crop and invert images for {}'.format(ids))
        error_list.append('{} Step 3'.format(ids))



    #Step 4+5. Get list of tif files in subjects directory
    #Get an image x and y dimensions for bounding box
    try:
        # print('step 4')
        list_of_files = natsort.natsorted(os.listdir(raw_single_tif_dir))
        list_of_files = [f for f in list_of_files if '.tif' in f]

        # print('step 5')
        img_for_shape = cv2.imread(os.path.join(raw_single_tif_dir,list_of_files[1]),cv2.IMREAD_UNCHANGED)
        inverted_height, inverted_width = img_for_shape.shape
        print(inverted_height,inverted_width)
    except:
        print('Unable to crop and invert images for {}'.format(ids))
        error_list.append('{} Steps 4/5'.format(ids))


    #step 6. make outdir for 3d chunks
    # print('step 6')
    if os.path.isdir(os.path.join(specimen_dir,'Chunks_of_{}'.format(chunk_size))):
        chunk_dir = os.path.join(specimen_dir,'Chunks_of_{}'.format(chunk_size))
    else:
        os.mkdir(os.path.join(specimen_dir,'Chunks_of_{}'.format(chunk_size)))
        chunk_dir = os.path.join(specimen_dir,'Chunks_of_{}'.format(chunk_size))

    #Step 7. Make each segment of (chunk_size) tif images into one 3d chunk
    try:
        stack_into_chunks(chunk_size,raw_single_tif_dir,chunk_dir,ids)
        chunk_err = None
        left_and_right = False

    except ValueError:
        #This will fail with a memory error. I.E. you were trying to make 3D chunks, larger than the amount of RAM available

        chunk_err = 1
        print('Error Generating Chunk Tiffs for {}'.format(ids))
        print("Will try splitting images in half")
        error_list.append('{} Value Error @ Step 7'.format(ids))
        left_and_right = True
        #Step 7.1 Delete 32 chunks directory and Make new directories for left and right
        try:
            shutil.rmtree(chunk_dir)

            #Make left and right directories
            raw_single_tif_dir_left = os.path.join(specimen_dir, 'Single_Tif_Images_Left')
            raw_single_tif_dir_right = os.path.join(specimen_dir,'Single_Tif_Images_Right')
            chunk_dir_left = os.path.join(specimen_dir,'Chunks_of_{}_Left'.format(chunk_size))
            chunk_dir_right = os.path.join(specimen_dir,'Chunks_of_{}_Right'.format(chunk_size))
            list_to_check = [raw_single_tif_dir_left,raw_single_tif_dir_right,chunk_dir_left,chunk_dir_right]
            for d in list_to_check:
                if not os.path.exists(d):
                    os.mkdir(d)

        except:
            print('Error Generating Left Right Dirs for {}'.format(ids))
            error_list.append('{}  Step 7.1'.format(ids))


        #Step 7.2 Check that when we divide by two (in half) we make two images with x-dim still a multiple of 64
        try:

            #Need to keep the x dimension of left and right tif files multiples of 64. This is only true if the width is
            # an EVEN multiple of 64
            #For example: lets say in step 3 we cropped the image to 6,464 (i.e. 64*101). Splitting this in half
            #will give 3,232 and 3,232 x-pixels in each left and right image. 3,232 is no longer a multiple of 64.
            #To solve this just use myround64 to find the closest 64x index and use that

            # print('Width/2 div 64 = {}'.format((inverted_width/2)/64))
            if ((int(inverted_width)/2)/64).is_integer():
                half_way_crop = int((inverted_width)/2)

            else:
                # print('NEED TO REINDEX FOR 64X')
                half_way_crop = int(myround64(inverted_width/2))

            #Loop through all individual tiffs and divide them
            for inverted_cropped_img in [tif_file for tif_file in natsort.natsorted(os.listdir(raw_single_tif_dir)) if '.tif' in tif_file]:
                processed_img_path = os.path.join(raw_single_tif_dir,inverted_cropped_img)
                processed_img = cv2.imread(processed_img_path,cv2.IMREAD_UNCHANGED)

                left_half = processed_img[0:int(inverted_height) , 0:half_way_crop]
                right_half = processed_img[0:int(inverted_height) , half_way_crop:int(inverted_width)]

                cv2.imwrite(os.path.join(raw_single_tif_dir_left,'Left_'+inverted_cropped_img), left_half)
                cv2.imwrite(os.path.join(raw_single_tif_dir_right,'Right_'+inverted_cropped_img), right_half)

            left_width = left_half.shape[1]
            right_width = right_half.shape[1]
        except:
            print('Error dividing tiffs into Left and Right images for {}'.format(ids))
            error_list.append('{} Step 7.2'.format(ids))

        #Step 7.3 Make Left and Right Chunks of 32
        try:
            #Now we can make our left and right chunks of 32
            # print('Making left and right chunks')
            stack_into_chunks(chunk_size,raw_single_tif_dir_left,chunk_dir_left,ids)
            stack_into_chunks(chunk_size,raw_single_tif_dir_right,chunk_dir_right,ids)
        except:
            print('Error Generating Left and Right Chunk Tiffs for {}'.format(ids))
            error_list.append('{} Step 7.3'.format(str(ids)))

    #Steps 8 and 9
    if chunk_err:

        #Step 8. Check the sizes of each chunk generated from step 7
        # print('step 8 chunk error')
        for chunky_directory in [chunk_dir_left,chunk_dir_right]:
            try:
                check_for_size_limit(chunky_directory)
            except:
                print('At least one of {} tif stacks exceeds 5Gb'.format(ids))
                error_list.append('{} Step 8'.format(ids))

        #Step 9. write bounding box for each side  (0 0 0 y x chunk_size)
        try:
            bound_box_left = [0,0,0,left_width,inverted_height,chunk_size]
            bb_dir_L = os.path.join(specimen_dir,'bbox_{}_Left.csv'.format(ids))
            pd.DataFrame({'bound_boxing':bound_box_left}).to_csv(bb_dir_L.format(ids))

            bound_box_right = [0,0,0,right_width,inverted_height,chunk_size]
            bb_dir_R = os.path.join(specimen_dir,'bbox_{}_Right.csv'.format(ids))
            pd.DataFrame({'bound_boxing':bound_box_right}).to_csv(bb_dir_R.format(ids))

        except:
            print('Unable to make bounding box for {}'.format(ids))
            error_list.append('{} Step 9'.format(ids))

    else:
        #Step 8. Check the sizes of each chunk generated from step 7
        # print('step 8 no chunk error')
        check_for_size_limit(chunk_dir)

        #Step 9. Generate The Bounding Box File (0 0 0 y x chunk_size)
        try:
            # print('bounding box step')
            bound_box = [0,0,0,inverted_width,inverted_height,chunk_size]
            bb_dir = os.path.join(specimen_dir,'bbox_{}.csv'.format(ids))
            pd.DataFrame({'bound_boxing':bound_box}).to_csv(bb_dir.format(ids))
        except:
            print('Unable to make bounding box for {}'.format(ids))
            error_list.append('{} Step 9'.format(ids))

    #make mip for single tiff files dir and then delete those dirs
    if left_and_right == True:
         raw_single_tif_dir_right = os.path.join(specimen_dir,'Single_Tif_Images_Right')
         mip_ofile_right = os.path.join(specimen_dir,'Single_Tif_Images_Right_Mip.tif')

         raw_single_tif_dir_left = os.path.join(specimen_dir,'Single_Tif_Images_Left')
         mip_ofile_left = os.path.join(specimen_dir,'Single_Tif_Images_Left_Mip.tif')

         dir_to_mip(indir = raw_single_tif_dir_right, ofile = mip_ofile_right )
         dir_to_mip(indir = raw_single_tif_dir_left, ofile = mip_ofile_left )

    else:
        mip_ofile = os.path.join(specimen_dir,"Single_Tif_Images_Mip.tif")
        dir_to_mip(indir=raw_single_tif_dir, ofile = mip_ofile)

    return error_list

def dir_to_mip(indir,ofile,mip_axis=2):
    """
    From a directory of single tif files, will create a maximum intensity projection (mip) along certain axis
    example: if mip_axis=2 creates xy mip
    """

    indir_files = os.listdir(indir)
    img_0_pth = os.path.join(indir,indir_files[0])
    img_0 = cv2.imread(img_0_pth,cv2.IMREAD_UNCHANGED)
    data_type = img_0.dtype
    # print(img_0.shape,len(indir_files))

    full_img = np.zeros((img_0.shape[0],img_0.shape[1],len(indir_files)))
    ct=-1
    for fn in indir_files:
        ct+=1
        pth = os.path.join(indir,fn)
        img = cv2.imread(pth,cv2.IMREAD_UNCHANGED)
        full_img[:,:,ct] = img

    mip_z_axis = np.max(full_img, axis=mip_axis).astype(data_type)
    imsave(ofile,mip_z_axis)


def main(specimen_id, raw_single_tif_dir, specimen_dir, invert_image_color, **kwargs):

    if not specimen_dir:
        specimen_dir = os.path.dirname(raw_single_tif_dir)

    returned_error_list = process_specimen(specimen_id,specimen_dir, raw_single_tif_dir, invert_image_color)
    if returned_error_list!=[]:
        print("error occured in preprocessing:")
        print(returned_error_list)

    else:
        print("Image Preprocessing Completed Without Error")

if __name__ == "__main__":
	module = ags.ArgSchemaParser(schema_type=InputSchema)
	main(**module.args)
