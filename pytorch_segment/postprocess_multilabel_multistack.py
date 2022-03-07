from multiprocessing import Pool
import numpy as np
import tifffile as tif
import os
import glob
import argparse
import natsort
import pandas as pd
from skimage.morphology import remove_small_objects, skeletonize_3d
from scipy import ndimage as ndi
from datetime import datetime
import torch

def load_stack(input_dir):
    # Load image stack filenames
    filelist = [f for f in os.listdir(input_dir) if f.endswith('.tif')] 
    filelist.sort()
    
    # Calculate stack size
    filename = os.path.join(input_dir, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = len(filelist), img.shape[0], img.shape[1]
        
    stack = np.zeros(cell_stack_size, dtype=np.uint8)
    for i, f in enumerate(filelist):
        filename = os.path.join(input_dir, f)
        img = tif.imread(filename)
        stack[i,:,:] = img
        
    return stack    

def postprocess(specimen_dir, indir, ids, error_list, threshold=0.3, size_threshold=2000, 
                max_stack_size=8000000000):
    try:
        # Step 1. Make postprocess output directory 
        indir_base_path = os.path.basename(indir)
            
        if 'Left' in indir_base_path:
            savedir = os.path.join(specimen_dir,'Left_Skeleton')       
        elif 'Right' in indir_base_path:
            savedir = os.path.join(specimen_dir,'Right_Skeleton')   
        else:
            savedir = os.path.join(specimen_dir,'Skeleton')
            
        if not os.path.isdir(savedir):
            os.mkdir(savedir)    
       
        # Step 2. Calculate soma centroid
        ch1_dir = os.path.join(indir, 'ch1')
        
        # Load stack
        stack = load_stack(ch1_dir)
        cell_stack_size = stack.shape
            
        # Save max intensity projection
        tif.imsave(os.path.join(specimen_dir,'MAX_' + indir.split('/')[-1] + '_ch1.tif'), 
                    np.max(stack, axis=0))

        # Zero values below threshold ~0.99
        stack[stack < 252] = 0 
                
        # Save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_ch1.csv'), np.stack((x,y,z,I), axis=1), 
                    fmt='%u', delimiter=',', header='x,y,z,I')
        
        # Calculate centroid and save as csv file
        cgx = np.sum(x*I)/np.sum(I)
        cgy = np.sum(y*I)/np.sum(I)
        cgz = np.sum(z*I)/np.sum(I)
        np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_soma_centroid.csv'), 
                    np.swapaxes(np.array([[cgx],[cgy],[cgz]]),0,1), fmt='%.1f', delimiter=',', header='x,y,z')
            
        # Step 3. Save axon/dendrites in csv format and max projections
        ch2_dir = os.path.join(indir, 'ch2')
        
        # Load stack
        stack = load_stack(ch2_dir)                    
            
        # Save max intensity projection
        tif.imsave(os.path.join(specimen_dir,'MAX_' + indir.split('/')[-1] + '_ch2.tif'), 
                    np.max(stack, axis=0))

        # Zero values below threshold
        stack[stack < int(np.round(255*threshold))] = 0
        
        # Save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_ch2.csv'), np.stack((x,y,z,I), axis=1), 
                    fmt='%u', delimiter=',', header='x,y,z,I')
        
        ch3_dir = os.path.join(indir, 'ch3')
                            
        # Load stack
        stack = load_stack(ch3_dir) 
            
        # Save max intensity projection
        tif.imsave(os.path.join(specimen_dir,'MAX_' + indir.split('/')[-1] + '_ch3.tif'), 
                    np.max(stack, axis=0))

        # Zero values below threshold
        stack[stack < int(np.round(255*threshold))] = 0
        
        # Save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_ch3.csv'), np.stack((x,y,z,I), axis=1), 
                    fmt='%u', delimiter=',', header='x,y,z,I')
        
        #Step 4. Calculate arbor segmentation
        # Create folder for arbor segmentation
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
            tif.imsave(os.path.join(ch5_dir,'%03d.tif'%i), img) 
            mask = np.zeros((img.shape), dtype=np.uint8)
            mask[img2 > img3] = 2
            mask[img3 > img2] = 3
            tif.imsave(os.path.join(mask_dir,'%03d.tif'%i), mask)        
        
        # Step 5. Postprocess arbor segmentation
        # Load image stack filenames
        filelist = [f for f in os.listdir(ch5_dir) if f.endswith('.tif')] 
        filelist.sort()
        
        cell_stack_memory = cell_stack_size[0]*cell_stack_size[1]*cell_stack_size[2]
        print('cell_stack_size (z,y,x):', cell_stack_size, cell_stack_memory)
        # If cell stack memory>max_stack_size (15GB for RAM=128GB), need to split
        num_parts = int(np.ceil(cell_stack_memory/max_stack_size))
        print('num_parts:', num_parts)
        
        # Split filelist
        idx = np.append(np.arange(0, cell_stack_size[0], int(np.ceil(cell_stack_size[0]/num_parts))), cell_stack_size[0]+1)
        print('idx:', idx)
        for i in range(num_parts):
            idx1 = idx[i]
            idx2 = idx[i+1]
            filesublist = filelist[idx1:idx2]
            print('part ', i, idx1, idx2, len(filesublist))

            # Load stack
            stack_size = len(filesublist), cell_stack_size[1], cell_stack_size[2]
            stack = np.zeros(stack_size, dtype=np.uint8)
            print('loading stack')
            for i, f in enumerate(filesublist):
                filename = os.path.join(ch5_dir, f)
                img = tif.imread(filename)
                stack[i,:,:] = img
            print(stack.shape, stack.dtype)

            print('removing smaller segments')
            # Binarize stack based on threshold
            stack = (stack > int(np.round(255*threshold))).astype(np.uint8)

            # Label connected components
            s = ndi.generate_binary_structure(3,3)
            stack = ndi.label(stack,structure=s)[0].astype(np.uint16)

            # Remove components smaller than size_threshold 
            # connectivity=3 - pixels are connected if their faces, edges, or corners touch
            stack = remove_small_objects(stack, min_size=size_threshold, connectivity=3)

            # Convert all connected component labels to 1
            stack = (stack > 0).astype(np.uint8)

            # Skeletonize stack
            print('skeletonizing stack')
            stack = skeletonize_3d(stack)

            # Save stack as multiple tif files
            print('saving stack')
            for i in range(stack.shape[0]):
                tif.imsave(os.path.join(savedir,'%03d.tif'%(i+idx1)), stack[i,:,:])

        # Save skeleton as csv file
        stack = load_stack(savedir)
            
        # Save nonzero pixels as csv file (x,y,z,I)
        z,y,x = np.nonzero(stack)
        I = stack[z,y,x]
        np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_skeleton.csv'), np.stack((x,y,z,I), axis=1), 
                    fmt='%u', delimiter=',', header='x,y,z,I')    
        
        # Step 6. Create labeled skeleton
        # Single and double cases
        if 'Left' in indir_base_path:
            pass       
        elif 'Right' in indir_base_path:
            # Process both left and right
            left_skeleton_file = os.path.join(specimen_dir, 'Left_Segmentation_skeleton.csv')
            right_skeleton_file = os.path.join(specimen_dir, 'Right_Segmentation_skeleton.csv')
            left_mask_dir = os.path.join(specimen_dir, 'Left_Segmentation', 'mask')
            right_mask_dir = os.path.join(specimen_dir, 'Right_Segmentation', 'mask')
            
            # Load skeleton
            df = pd.read_csv(left_skeleton_file)
            arbor_l = df.values[:,0:3]
            df = pd.read_csv(right_skeleton_file)
            arbor_r = df.values[:,0:3] 
            
            # Load mask
            stack = load_stack(left_mask_dir)
            mask_l_shape = stack.shape

            # Use mask to find node type
            node_type_l = stack[arbor_l[:,2], arbor_l[:,1], arbor_l[:,0]]
            node_type_l[node_type_l==0] = 5

            # Load mask
            stack = load_stack(right_mask_dir)

            # Use mask to find node type
            node_type_r = stack[arbor_r[:,2], arbor_r[:,1], arbor_r[:,0]]
            node_type_r[node_type_r==0] = 5
            
            arbor_r[:,0] = arbor_r[:,0] + mask_l_shape[2] # shift x coordinate
            arbor = np.concatenate((arbor_l, arbor_r), axis=0)
            node_type = np.concatenate((node_type_l, node_type_r), axis=0)
            
            # Save labeled skeleton
            np.savetxt(os.path.join(specimen_dir, 'Segmentation_skeleton_labeled.csv'), np.column_stack((arbor,node_type)), 
                       fmt='%u', delimiter=',', header='x,y,z,type')
        else:
            # Process single
            skeleton_file = os.path.join(specimen_dir, 'Segmentation_skeleton.csv')
            mask_dir = os.path.join(indir, 'mask')
            
            # Load skeleton
            df = pd.read_csv(skeleton_file)
            arbor = df.values[:,0:3]
            
            # Load mask
            stack = load_stack(mask_dir)
            
            # Use mask to find node type
            node_type = stack[arbor[:,2], arbor[:,1], arbor[:,0]]
            node_type[node_type==0] = 5
            
            # Save labeled skeleton
            np.savetxt(os.path.join(specimen_dir, indir.split('/')[-1] + '_skeleton_labeled.csv'), np.column_stack((arbor,node_type)), 
                       fmt='%u', delimiter=',', header='x,y,z,type')
    except:
        print('error with postprocess')
        error_list.append(str(ids)+ ' -postprocess')
    return error_list


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-v', type=str, help='directory of validation/test data')
    parser.add_argument('--csv', type=str, help='input csv with specimen id column header')
    parser.add_argument('--num_processes', type=int, default = 1, help='number of processes to run parallel')
    
    today_time = datetime.today()
    date_time = '_'.join([today_time.strftime("%b_%d_%Y"), str(today_time.hour), str(today_time.minute), 
                      str(today_time.second)])

    args = parser.parse_args()
    outdir = args.outdir
    df = pd.read_csv(args.csv)
    specimens = list(df.specimen_id.values)

    num_processes = args.num_processes
    
    all_error_list = []
    if num_processes == 1:

        for sp_ids in specimens:       
            specimen_dir = os.path.join(outdir,str(sp_ids))
            error_list = []
            error_list_2 = []

            if os.path.exists(os.path.join(specimen_dir,'Left_Segmentation')):
                # segmentaion dir 
                indir_left = os.path.join(specimen_dir,'Left_Segmentation')
                indir_right = os.path.join(specimen_dir,'Right_Segmentation')
                print('indir:', indir_left, indir_right)
                
                # Postprocess
                res_l = postprocess(specimen_dir, indir_left, sp_ids, error_list)
                res_r = postprocess(specimen_dir, indir_right, sp_ids, error_list_2)
                
                all_error_list.append(res_l)
                all_error_list.append(res_r)
                
            else:
                indir = os.path.join(specimen_dir,'Segmentation')
                print('indir:', indir)
                
                # Postprocess
                res = postprocess(specimen_dir, indir, sp_ids, error_list)
                all_error_list.append(res)

    else:
        p = Pool(processes=num_processes)
        parallel_input = [(int(i), error_list) for i in specimens] 
        all_error_list = p.starmap(postprocess, parallel_input)

    with open('{}_postprocess_error_log.txt'.format(date_time), 'w') as f:
        for item in all_error_list:
            f.write("%s\n" % item)
