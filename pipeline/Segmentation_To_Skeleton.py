import numpy as np
import tifffile as tif
import os
import natsort
import pandas as pd
from skimage.morphology import remove_small_objects, skeletonize_3d
from scipy import ndimage as ndi
import argschema as ags


class InputSchema(ags.ArgSchema):
    specimen_dir = ags.fields.InputDir(description=" ")
    specimen_id = ags.fields.Str(default=None,description=" ")
    intensity_threshold = ags.fields.Int(default=50, description = "50 for spiny 252 for aspiny")

def load_stack(input_dir):
    """
    Load files into numpy array

    :param input_dir: input directory with single tif images
    :return: 3d numpy array with dimensions (y,x,z)

    """
    # load image stack filenames
    filelist = [f for f in os.listdir(input_dir) if f.endswith('.tif')] 
    filelist = natsort.natsorted(filelist)
    
    # calculate stack size
    filename = os.path.join(input_dir, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = len(filelist), img.shape[0], img.shape[1]
        
    stack = np.zeros(cell_stack_size, dtype=np.uint8)
    for i, f in enumerate(filelist):
        filename = os.path.join(input_dir, f)
        img = tif.imread(filename)
        stack[i,:,:] = img
        
    return stack    

def postprocess(specimen_dir, indir, ids, error_list, intensity_threshold, threshold=0.3, size_threshold=2000, 
                max_stack_size=8000000000):
    """
    This will "clean up" segmentation files, removing small, low intensity segments and converting segmentation images
    into lower size csv files (with columns = x,y,z,intensity,label) so that, once the large segmentation images are
    removed, the raw segmentation is still available in low memory form. This script will also generate maximum
    intensity projections which can be useful for further processing the swc file.

    This will create a directory called /Skeleton in the specimen dir, which is what we call a post-processed version
    of the segmentation.

    :param specimen_dir: specimens directory
    :param indir: which directory we are post processing (i.e. Channel 1,2 or 3)
    :param ids: specimen id
    :param error_list: empty list that will be populated if errors occur
    :param intensity_threshold:
    :param threshold:
    :param size_threshold:
    :param max_stack_size: maximum size (in bytes) to load into memory at once
    :return:
    """
    print("Starting to postprocess segmentation")
    try:
        #Step 1. Make postprocess output directory 
        # try:
        indir_base_path = os.path.basename(indir)
            
        if 'Left' in indir_base_path:
            savedir = os.path.join(specimen_dir,'Left_Skeleton')       
        elif 'Right' in indir_base_path:
            savedir = os.path.join(specimen_dir,'Right_Skeleton')   
        else:
            savedir = os.path.join(specimen_dir,'Skeleton')
            
        if not os.path.isdir(savedir):
            os.mkdir(savedir)    


        #Step 2. Calculate soma centroid
        # try:
        ch1_dir = os.path.join(indir, 'ch1')
        
        # load stack
        stack = load_stack(ch1_dir)
        cell_stack_size = stack.shape
            
        # save max intensity projection
        tif.imsave(os.path.join(specimen_dir,'MAX_' + indir.split('/')[-1] + '_ch1.tif'), 
                    np.max(stack, axis=0))
        
        # #Save other MIP views
        # tif.imsave(os.path.join(specimen_dir,'MAX_xz_' + indir.split('/')[-1] + 'axis1_ch1.tif'), 
        #             np.max(stack, axis=1))

        #Save other MIP views
        tif.imsave(os.path.join(specimen_dir,'MAX_yz_' + indir.split('/')[-1] + '_ch1.tif'), 
                    np.max(stack, axis=2))

        # zero values below threshold ~0.99
        #stack = (stack > 251).astype(np.uint8) # binarize stack based on threshold ~0.99
        stack[stack < intensity_threshold] = 0
                
        # save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, '{}_ch1.csv'.format(indir_base_path)), np.stack((x,y,z,I), axis=1),
                    fmt='%u', delimiter=',', header='x,y,z,I')
        
        # calculate centroid and save as csv file
        cgx = np.sum(x*I)/np.sum(I)
        cgy = np.sum(y*I)/np.sum(I)
        cgz = np.sum(z*I)/np.sum(I)
        np.savetxt(os.path.join(specimen_dir, '{}_soma_centroid.csv'.format(indir_base_path)),
                    np.swapaxes(np.array([[cgx],[cgy],[cgz]]),0,1), fmt='%.1f', delimiter=',', header='x,y,z')
            
        # except:
        #     print('error with soma extraction')
        #     error_list.append(str(ids)+ ' -soma') 
            
        #Step 3. Save axon/dendrites in csv format and max projections
        # try:
        ch2_dir = os.path.join(indir, 'ch2')
        
        # load stack
        stack = load_stack(ch2_dir)                    
            
        # save max intensity projection
        tif.imsave(os.path.join(specimen_dir,'MAX_{}_ch2.tif'.format(indir_base_path)),
                    np.max(stack, axis=0))

        # zero values below threshold
        stack[stack < int(np.round(255*threshold))] = 0
        
        # save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, '{}_ch2.csv'.format(indir_base_path)), np.stack((x,y,z,I), axis=1),
                    fmt='%u', delimiter=',', header='x,y,z,I')
        
        ch3_dir = os.path.join(indir, 'ch3')
                            
        # load stack
        stack = load_stack(ch3_dir) 
            
        # save max intensity projection
        tif.imsave(os.path.join(specimen_dir,'MAX_' + indir.split('/')[-1] + '_ch3.tif'), 
                    np.max(stack, axis=0))

        # zero values below threshold
        stack[stack < int(np.round(255*threshold))] = 0
        
        # save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_ch3.csv'), np.stack((x,y,z,I), axis=1), 
                    fmt='%u', delimiter=',', header='x,y,z,I')
            
        # except:
        #     print('error with axon dendrite')
        #     error_list.append(str(ids)+ ' -axon_dendrite') 
        
        #Step 4. Calculate arbor segmentation
        # try:
        # create folder for arbor segmentation
        ch5_dir = os.path.join(indir, 'ch5')
        if not os.path.isdir(ch5_dir):
            os.mkdir(ch5_dir) 
        
        mask_dir = os.path.join(indir, 'mask')
        if not os.path.isdir(mask_dir):
            os.mkdir(mask_dir)
        
        filelist = [f for f in os.listdir(ch2_dir) if f.endswith('.tif')] 
        filelist.sort()
        
        for i, f in enumerate(filelist):
            filename = os.path.join(ch2_dir, f)
            img2 = tif.imread(filename)
            filename = os.path.join(ch3_dir, f)
            img3 = tif.imread(filename)
            img = np.maximum(img2, img3)
            tif.imsave(os.path.join(ch5_dir,'%03d.tif'%i), img) # consecutive numbers
            mask = np.zeros((img.shape), dtype=np.uint8)
            mask[img2 > img3] = 2
            mask[img3 > img2] = 3
            tif.imsave(os.path.join(mask_dir,'%03d.tif'%i), mask) # consecutive numbers       
        # except:
        #     print('error with arbor')
        #     error_list.append(str(ids)+ ' -arbor') 
        
        #Step 5. Postprocess arbor segmentation
        # try:
        # load image stack filenames (consecutive numbers)
        filelist = [f for f in os.listdir(ch5_dir) if f.endswith('.tif')] 
        filelist.sort()
        
        cell_stack_memory = cell_stack_size[0]*cell_stack_size[1]*cell_stack_size[2]
        print('cell_stack_size (z,y,x):', cell_stack_size, cell_stack_memory)
        # if cell stack memory>max_stack_size (15GB for RAM=128GB), need to split
        num_parts = int(np.ceil(cell_stack_memory/max_stack_size))
        print('num_parts:', num_parts)
        
        # split filelist
        idx = np.append(np.arange(0, cell_stack_size[0], int(np.ceil(cell_stack_size[0]/num_parts))), cell_stack_size[0]+1)
        print('idx:', idx)
        for i in range(num_parts):
            idx1 = idx[i]
            idx2 = idx[i+1]
            filesublist = filelist[idx1:idx2]
            print('part ', i, idx1, idx2, len(filesublist))

            # load stack
            stack_size = len(filesublist), cell_stack_size[1], cell_stack_size[2]
            stack = np.zeros(stack_size, dtype=np.uint8)
            print('loading stack')
            for i, f in enumerate(filesublist):
                filename = os.path.join(ch5_dir, f)
                img = tif.imread(filename)
                stack[i,:,:] = img
            print(stack.shape, stack.dtype)

            print('removing smaller segments')
            # binarize stack based on threshold
            stack = (stack > int(np.round(255*threshold))).astype(np.uint8)

            # label connected components
            s = ndi.generate_binary_structure(3,3)
            stack = ndi.label(stack,structure=s)[0].astype(np.uint16)

            # remove components smaller than size_threshold 
            # connectivity=3 - pixels are connected if their faces, edges, or corners touch
            stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)

            # convert all connected component labels to 1
            stack = (stack > 0).astype(np.uint8)

            # skeletonize stack
            print('skeletonizing stack')
            stack = skeletonize_3d(stack)

            # save stack as multiple tif files
            print('saving stack')
            for i in range(stack.shape[0]):
                tif.imsave(os.path.join(savedir,'%03d.tif'%(i+idx1)), stack[i,:,:])

        # save skeleton as csv file
        stack = load_stack(savedir)
            
        # save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, '{}_skeleton.csv'.format(indir_base_path)), np.stack((x,y,z,I), axis=1),
                    fmt='%u', delimiter=',', header='x,y,z,I')    
        # except:
        #     print('error with skeleton')
        #     error_list.append(str(ids)+ ' -skeleton')
        
        #Step 6. Create labeled skeleton
        # single and double cases
        if 'Left' in indir_base_path:
            pass       
        elif 'Right' in indir_base_path:
            # process both left and right
            left_skeleton_file = os.path.join(specimen_dir, 'Left_Segmentation_skeleton.csv')
            right_skeleton_file = os.path.join(specimen_dir, 'Right_Segmentation_skeleton.csv')
            left_mask_dir = os.path.join(specimen_dir, 'Left_Segmentation', 'mask')
            right_mask_dir = os.path.join(specimen_dir, 'Right_Segmentation', 'mask')
            
            # load skeleton
            df = pd.read_csv(left_skeleton_file)
            arbor_l = df.values[:,0:3]
            df = pd.read_csv(right_skeleton_file)
            arbor_r = df.values[:,0:3] 
            
            # load mask
            stack = load_stack(left_mask_dir)
            mask_l_shape = stack.shape

            # use mask to find node type
            node_type_l = stack[arbor_l[:,2], arbor_l[:,1], arbor_l[:,0]]
            node_type_l[node_type_l==0] = 5

            # load mask
            stack = load_stack(right_mask_dir)

            # use mask to find node type
            node_type_r = stack[arbor_r[:,2], arbor_r[:,1], arbor_r[:,0]]
            node_type_r[node_type_r==0] = 5
            
            arbor_r[:,0] = arbor_r[:,0] + mask_l_shape[2] # shift x coordinate
            arbor = np.concatenate((arbor_l, arbor_r), axis=0)
            node_type = np.concatenate((node_type_l, node_type_r), axis=0)
            
            # save labeled skeleton
            np.savetxt(os.path.join(specimen_dir, 'Segmentation_skeleton_labeled.csv'), np.column_stack((arbor,node_type)), 
                       fmt='%u', delimiter=',', header='x,y,z,type')
        else:
            # process single
            skeleton_file = os.path.join(specimen_dir, 'Segmentation_skeleton.csv')
            mask_dir = os.path.join(indir, 'mask')
            
            # load skeleton
            df = pd.read_csv(skeleton_file)
            arbor = df.values[:,0:3]
            
            # load mask
            stack = load_stack(mask_dir)
            
            # use mask to find node type
            node_type = stack[arbor[:,2], arbor[:,1], arbor[:,0]]
            node_type[node_type==0] = 5
            
            # save labeled skeleton
            np.savetxt(os.path.join(specimen_dir, '{}_skeleton_labeled.csv'.format(indir_base_path)), np.column_stack((arbor,node_type)),
                       fmt='%u', delimiter=',', header='x,y,z,type')

        #
        # directories_to_remove = ["Chunks_of_32","Chunks_of_32_Left","Chunks_of_32_Right",
        #                     "Segmentation","Segmentation_Left","Segmentation_Right",
        #                     "Single_Tif_Images","Single_Tif_Images_Left","Single_Tif_Images_Right"]
        # print("Cleaning Up:")
        # for dir_name in directories_to_remove:
        #     full_dir_name = os.path.join(specimen_dir,dir_name)
        #     if os.path.exists(full_dir_name):
        #         print(full_dir_name)
        #         shutil.rmtree(full_dir_name)

    except:
        print('error with postprocess')
        error_list.append(str(ids)+ ' -postprocess')

    return error_list


def main(specimen_dir,specimen_id,intensity_threshold,**kwargs):

    error_list = []
    error_list_2 = []

    indir_left = os.path.join(specimen_dir, 'Left_Segmentation')
    if os.path.exists(indir_left):
        indir_right = os.path.join(specimen_dir,'Right_Segmentation')
        print('indir:', indir_left, indir_right)

        res_l = postprocess(specimen_dir, indir_left, specimen_id, error_list,intensity_threshold)
        res_r = postprocess(specimen_dir, indir_right, specimen_id, error_list_2,intensity_threshold)


    else:
        indir = os.path.join(specimen_dir,'Segmentation')
        print('indir:', indir)

        # postprocess
        res = postprocess(specimen_dir, indir, specimen_id, error_list, intensity_threshold)



if __name__ == "__main__":
	module = ags.ArgSchemaParser(schema_type=InputSchema)
	main(**module.args)
