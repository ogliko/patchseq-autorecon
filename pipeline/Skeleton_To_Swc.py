import numpy as np
import pandas as pd 
from collections import deque
import tifffile as tif
import itertools
from operator import add
import os
import argschema as ags
from collections import defaultdict
from tifffile import imsave
import cv2
import shutil
from scipy.ndimage import label, morphology, generate_binary_structure
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import math

class LayerSchema(ags.ArgSchema):
    specimen_id = ags.fields.Int(description='specimen id')
    specimen_dir = ags.fields.InputDir(description = 'Specimens Directory')
    remove_intermediate_files = ags.fields.Boolean(default=False,description='as this is the last step of pipeline, this flag will remove all intermediate images generated')
    max_stack_size = ags.fields.Int(default = 7000000000, description='maximum size of image stack to load at once (in bytes)')
    minimum_soma_area_pixels = ags.fields.Int(default=500,description='minimum xy area (in pixels^2) of expected soma. This is used in identifying the soma from channel 1 segmentation')
    soma_connection_threshold = ags.fields.Int(default=100,description='all individual connected components within this distance from the soma centroid will be connected as a soma stem')

def stitch_skeleton_dir(specimen_dir,skeleton_dir):
    """
    If in the first step, PreProcess_ImageStack.py, we needed to split the image into left and right halves due to
    restricted memory resources, we will now stitch the left and right Skeleton directory (generated in 
    Segmentation_TO_Skeleton.py) into one skeleton directory. 
    
    :param specimen_dir:specimen directory
    :param skeleton_dir: directory to put stitched skeleton images
    :return:
    """
    left_skeleton_dir = os.path.join(specimen_dir,'Left_Skeleton')
    right_skeleton_dir = os.path.join(specimen_dir,'Right_Skeleton')

    left_files = [f for f in os.listdir(left_skeleton_dir) if f.endswith('.tif')] 
    left_files.sort()
    right_files = [f for f in os.listdir(right_skeleton_dir) if f.endswith('.tif')] 
    right_files.sort()

    for img in left_files:
        left_img_path = os.path.join(left_skeleton_dir,img)
        right_img_path = os.path.join(right_skeleton_dir,img)
        
        left_img = cv2.imread(left_img_path,cv2.IMREAD_UNCHANGED)
        right_img = cv2.imread(right_img_path,cv2.IMREAD_UNCHANGED)
        combined_img = np.append(left_img,right_img,axis=1)
        imsave(os.path.join(skeleton_dir,img),combined_img)

def assign_parent_child_relation(start_node, start_nodes_parent, parent_dict, neighbors_dict):
    """
    This function uses breadth first traversal to assign parent child relationships in a unlabeled neurite segment.

    Keys in the neighbors_dict are the coordinates of the nodes in this connected component. Each coordinate has at
    least one neighbor (by definition of the connected-components algorithm). The value for a given coordinate in
    neighbors_dict is a list of that coordinates neighboring node(s) as defined by 3d connectivity in
    connected components algorithm.

    The goal of this function is to update the parents_dict so that we have relationship structure to our swc graph.

    :param start_node: the leaf node (coordinate with only one neighbor) that is closest to the soma
    :param start_nodes_parent: if the start node is close enough, it's parent will be the soma, otherwise
                                it's parent is -1 and is treated as disconnected from the soma
    :param parent_dict: dictionary containing each coordinates parent node
    :param neighbors_dict: dictionary with keys as 3-d coorindate tuple and values are list of coordinates
    :return: None
    """
    parent_dict[start_node] = start_nodes_parent
    queue = deque([start_node])
    while len(queue) > 0:
        current_node = queue.popleft()
#         print('')
#         print('current node {} {}'.format(node_dict[current_node], current_node))
        my_connections = neighbors_dict[current_node]
#         [print('my connections {} {}'.format(node_dict[ss], ss)) for ss in my_connections]
        for node in my_connections:
#             print('checking node {}'.format(node_dict[node]))
            if node not in parent_dict:
#                 print('Assigning node {} to be the child of {}'.format(node_dict[node],node_dict[current_node]))
                parent_dict[node] = current_node
                queue.append(node)
            else:
                p = 'Initial start node' if parent_dict[node] == start_nodes_parent else str([parent_dict[node]])
#                 print('{} already has a parent {}'.format(node_dict[node], p))


def consolidate_conn_components(ccs_to_merge):
    """
    Usually we are unable to load the entire Skeleton image stack into memory at once. Therefore, we
    have to do this in chunks. But when going from skeleton image stack to swc, we don't want our swc file to have
    gaps/slices at these chunk indices. This function will take a dictionary of coordinates across these chunk
    indices and stitch them together as to reconnect neuron segments that were disrupted by our memory limited chunking.

    :param ccs_to_merge: a dictionary where keys are a connected component label and values are a set of all coordinates
                         that will need to be merged into the key connected component.
    :return: my_dict: a dictionary that has new neighbor relationships for each coordinate of the connected components
    """
    pre_qc = set()
    all_nodes={}
    for k,v in ccs_to_merge.items():
        pre_qc.add(k)
        all_nodes[k] = set()
        for vv in v:
            pre_qc.add(vv)
            all_nodes[vv] = set()

    nodes_to_remove = set()
    for start_node in ccs_to_merge.keys():
        rename_count = 0
        # print('Start Node {} assignment = {}'.format(start_node, all_nodes[start_node]))
        if all_nodes[start_node] == set():
            # print('Starting at {}'.format(start_node))
            queue = deque([start_node])
            start_value = start_node
            back_track_log = set()
            while len(queue) > 0:

                current_node = queue.popleft()

                if all_nodes[current_node] == set():
                    # print('assigning current node {} to {}'.format(current_node,start_value))
                    back_track_log.add(current_node)
                    all_nodes[current_node].add(start_value)
                    if current_node in ccs_to_merge.keys():
                        # print('appending children to the queue')
                        [queue.append(x) for x in ccs_to_merge[current_node]]
                else:
                    rename_count +=1 
                    if rename_count <2:
                        nodes_to_remove.add(start_node)
                        # print('Backtracking when i got to node {}'.format(current_node))
                        start_value = next(iter(all_nodes[current_node]))
                        # print('Updating the value to be {}'.format(start_value))
                        for node in back_track_log:
                            # print('Going back to update node {} to {}'.format(node,start_value))
                            all_nodes[node].clear()
                            all_nodes[node].add(start_value)

                    else: 
                        #need to remove all nodes that have this already assigned value
                        #and updated them to the start_value assigned on line 36  
                        value_to_remove = next(iter(all_nodes[current_node]))
                        # print('Found {} already labeled node at {}. Its label = {}'.format(rename_count,current_node,value_to_remove))
                        nodes_to_push_rename = [k for k,v in all_nodes.items() if value_to_remove in v]

                        for node in nodes_to_push_rename:
                            all_nodes[node].clear()
                            all_nodes[node].add(start_value)    
                            if node in ccs_to_merge.keys(): #cant remove it from the keys if its a leaf
                                nodes_to_remove.add(start_node)


    my_dict = defaultdict(set)
    for k,v in all_nodes.items():
        if k !=next(iter(v)):
            my_dict[next(iter(v))].add(k)

    post_qc = set()
    for k,v in my_dict.items(): 
        post_qc.add(k)
        for vv in v:
            post_qc.add(vv)

    # # Sanity Check
    # for i in pre_qc:
    #     if i not in post_qc:
    #         print('WARNING: ISSUE MERGING CONNECTED COMPONENTS. Node {} is in the input dict but not output'.format(i))

    return my_dict
    

#Processing Functions
def skeleton_to_swc_parallel(sp_id,specimen_dir,remove_intermediate_files,max_stack_size, minimum_soma_area_pixels, soma_connection_threshold):
    """
    Worker function that will take a specimen directory, using the skeleton directory, and other files that have been
    generated through this pipeline and convert the skeleton into an swc file. The swc file will have a node
    at every voxel that is deemed signal.


    :param sp_id: specimen id
    :param specimen_dir: specimen dictory
    :param remove_intermediate_files: if true, remove all the image files generated throughout pipeline
    :param max_stack_size: maximum stack size to load at once (bytes)
    :param minimum_soma_area_pixels: minimum xy area (in pixels^2) of expected soma. This is used in identifying the
                                    soma from channel 1 segmentation
    :param soma_connection_threshold: maximum distance from soma a segment will be consider a stem from the soma
    :return: None
    """

    print('Starting To Process {}'.format(sp_id))
    skeleton_dir = os.path.join(specimen_dir,'Skeleton')

    #Stitch together skeleton images when necessary
    skeleton_dir_L = os.path.join(specimen_dir,'Left_Skeleton')
    if os.path.exists(skeleton_dir) == False:
        os.mkdir(skeleton_dir)

    if os.path.exists(skeleton_dir_L) == True:
        #check if there is already a skeleton dir
        if os.path.exists(skeleton_dir):
            #make sure all the files are in skeleton dir
            if len([f for f in os.listdir(skeleton_dir) if '.tif' in f]) != len([f for f in os.listdir(skeleton_dir_L) if '.tif' in f]):
                 stitch_skeleton_dir(specimen_dir,skeleton_dir)
        else:
            stitch_skeleton_dir(specimen_dir,skeleton_dir)


    # Load centroid file, check if left right division is present. if it is check for scenario where left centroid
    # file is empty and right is not. need to add to the x value of the centroid because the image was split. X
    # Coordinates in the right image will need to be increased by half of the whole image x-size
    centroid_file = os.path.join(specimen_dir,'Segmentation_soma_centroid.csv')

    left_segmentation_file = os.path.join(specimen_dir,'Left_Segmentation_soma_centroid.csv')
    if os.path.exists(left_segmentation_file):
        temp_img = tif.imread(os.path.join(skeleton_dir_L,'001.tif'))
        left_df = pd.read_csv(os.path.join(specimen_dir,'Left_Segmentation_soma_centroid.csv'))
        right_df = pd.read_csv(os.path.join(specimen_dir,'Right_Segmentation_soma_centroid.csv'))
        if left_df.empty and not right_df.empty:
            right_df['# x'] = right_df['# x'] + int(temp_img.shape[1])    
            right_df.to_csv(os.path.join(specimen_dir,'Right_Segmentation_soma_centroid.csv'))
            centroid_file = os.path.join(specimen_dir,'Right_Segmentation_soma_centroid.csv')
        elif not left_df.empty and right_df.empty:
            centroid_file = os.path.join(specimen_dir,'Left_Segmentation_soma_centroid.csv')
        else:
            #theyre both empty or theyre both full, assume the latter
            right_df['# x'] = right_df['# x'] + int(temp_img.shape[1])
            left_df = left_df.append(right_df)
            pd.DataFrame(data=left_df.mean(axis=0)).T.to_csv(centroid_file) 
    print('Centroid File = {}'.format(centroid_file))


    #Make sure the soma coords file exists, stitch together if needed
    soma_coords_file = os.path.join(specimen_dir,'Segmentation_ch1.csv')
    split_side_file = os.path.join(specimen_dir,'Left_Segmentation_ch1.csv')

    if os.path.exists(split_side_file):
        print('stitching together the ch1 Left+Right segmentation csvs')
        temp_img = tif.imread(os.path.join(skeleton_dir_L,'001.tif'))
        ch1_Left_df = pd.read_csv(os.path.join(specimen_dir,'Left_Segmentation_ch1.csv'))
        ch1_Right_df = pd.read_csv(os.path.join(specimen_dir,'Right_Segmentation_ch1.csv'))
        ch1_Right_df['# x'] = ch1_Right_df['# x']+ int(temp_img.shape[1])
        ch1_Left_df.append(ch1_Right_df).to_csv(soma_coords_file,index=False)

    #Should alway exists, should not be left and right 
    skeleton_labels_file = os.path.join(specimen_dir,'Segmentation_skeleton_labeled.csv')
    print('Skeleton Dir = {}'.format(skeleton_dir))

    #Calculate how many files to load as not to exceed memory limit per iteration 
    filelist = [f for f in os.listdir(skeleton_dir) if f.endswith('.tif')] 
    filelist.sort()
    filename = os.path.join(skeleton_dir, filelist[0])
    img = tif.imread(filename)
    cell_stack_size = len(filelist), img.shape[0], img.shape[1]
    cell_stack_memory = cell_stack_size[0]*cell_stack_size[1]*cell_stack_size[2]
    print('cell_stack_size (z,y,x,memory): ({},{})'.format(cell_stack_size,cell_stack_memory))
    # if cell stack memory>max_stack_size need to split
    num_parts = int(np.ceil(cell_stack_memory/max_stack_size))
    print('num_parts:', num_parts)

    idx = np.append(np.arange(0, cell_stack_size[0], int(np.ceil(cell_stack_size[0]/num_parts))), cell_stack_size[0]+1)
    shared_slices = idx[1:-1]
    both_sides_of_slices = np.append(shared_slices,shared_slices-1)

    #Initialize variables before begining chunk look
    connected_components_on_border = {}
    for j in both_sides_of_slices:
        connected_components_on_border[j] = []
    previous_cc_count = 0
    slice_count = 0
    full_neighbors_dict = {}
    node_component_label_dict = {}
    cc_dict = {}

    # For each chunk of our image stack we will load it and run connected components. For each connected component, we
    # keep track of the component id, and the X,Y and Z coordinates that comprise this component
    for i in range(num_parts):
        print('At Part {}'.format(i))
        idx1 = idx[i]
        idx2 = idx[i+1]
        filesublist = filelist[idx1:idx2]
        # print('part ', i, idx1, idx2, len(filesublist))

        # load stack and run connected components
        cv_stack = []
        for i, f in enumerate(filesublist):
            filename = os.path.join(skeleton_dir, f)
            img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
            cv_stack.append(img)
        three_d_array = np.stack(cv_stack)

        # labels_out = cc3d.connected_components(three_d_array, connectivity=26) #a 3d matrix where pixel values = connected component labels

        struct = generate_binary_structure(3,3)
        labels_out,_ = label(three_d_array,structure=struct)

        current_number_of_components = np.max(labels_out)
        # print('There are {} CCs in this stack of images'.format(current_number_of_components))
        
        #Create range for connected components across all chunks of image stack
        if previous_cc_count == 0:
            cc_range = range(1,current_number_of_components+1)
        else:
            cc_range = range(previous_cc_count+1,previous_cc_count+1+current_number_of_components)

        for cc in cc_range:
            cc_dict[cc] = {'X':[],'Y':[],'Z':[]}
        
        #Load each slice of our connected component array so that we can get coordinates to the connected components
        for single_image in labels_out:
            single_image_unique_labels = np.unique(single_image) #return indices and ignore 0 
            for unique_label in single_image_unique_labels:
                if unique_label != 0:
                    indices = np.where(single_image==unique_label)
                    conn_comp_apriori_num = unique_label+previous_cc_count
                    [cc_dict[conn_comp_apriori_num]['Y'].append(coord) for coord in indices[0]]
                    [cc_dict[conn_comp_apriori_num]['X'].append(coord) for coord in indices[1]]
                    [cc_dict[conn_comp_apriori_num]['Z'].append(x) for x in [slice_count]*len(indices[1])]   

                    if slice_count in both_sides_of_slices:
                        connected_components_on_border[slice_count].append(conn_comp_apriori_num)

            slice_count+=1

    ################################################################################################################
    # Iterate through this chunks' connected components and update entire image stack neighbors dictionary
    ################################################################################################################
   
        for conn_comp in cc_range:
            # print('Analyzing Conn Component {}'.format(conn_comp))
            coord_values = cc_dict[conn_comp]
            component_coordinates = np.array([coord_values['X'],coord_values['Y'],coord_values['Z']]).T

            # Making a node dictionary for this con comp so we can lookup in the 26 node check step
            node_dict = {}
            count=0
            for c in component_coordinates:
                count+=1
                node_dict[tuple(c)] = count
                node_component_label_dict[tuple(c)] = conn_comp


            #26 nodes to check in defining neighbors dict
            # to find each nodes' "connection" we need to check all possible movements that are accepted in 3d
            # connected components. Imagine a 3x3x3 cube centered on a given node, any other voxel in that
            # cube would indicate a connection between the current centered node.
            movement_vectors = ([p for p in itertools.product([0,1,-1], repeat=3)])
            neighbors_dict = {}
            for node in component_coordinates:

                node_neighbors = [] 
                for vect in movement_vectors:
                    node_to_check = tuple(list(map(add,tuple(node),vect)))
                    if node_to_check in node_dict.keys():
                        node_neighbors.append(node_to_check)

                #remove myself from my node neightbors list
                node_neighbors = set([x for x in node_neighbors if x != tuple(node)])
                neighbors_dict[tuple(node)] = node_neighbors
                full_neighbors_dict[conn_comp] = neighbors_dict


        previous_cc_count += current_number_of_components

    # print('Finished Loading all chunks of skeleton and running cc on them at = {}'.format(datetime.now()))
    ################################################################################################################
    # All image chunks have been loaded and full neighbors dict is constructed. We need to now merge connected components
    # across the chunk indices. To do so, find nodes on left and right of the slice whos Z index == slice edge.
    # here "left" and "right" can be thought of as -Z and +Z relative to a given chunk index
    # ###############################################################################################################
    print('Merging Conn Components across chunk indices')
    # Initializing Nodes on either side of slice boundary
    nodes_to_left_of_boundary = {}
    for x in shared_slices-1:
        nodes_to_left_of_boundary[x] = defaultdict(list)

    nodes_to_right_of_boundary = {}
    for x in shared_slices:
        nodes_to_right_of_boundary[x] = defaultdict(list)
        
        
    #assigning nodes only with z value on edge to left or right side 
    for key,val in connected_components_on_border.items():
        for con_comp_label in val:
            coord_values = full_neighbors_dict[con_comp_label].keys()
            for coord in coord_values:
                z = coord[-1]
                if z == key:
                    if z in shared_slices-1:
                        nodes_to_left_of_boundary[key][con_comp_label].append(tuple(coord))
                    else:
                        nodes_to_right_of_boundary[key][(tuple(coord))] = con_comp_label   
            #Redundancy
            # coord_values = cc_dict[con_comp_label] 
            # component_coordinates = np.array([coord_values['X'],coord_values['Y'],coord_values['Z']]).T
            # for coord in component_coordinates:
            #     z = coord[-1]
            #     if z == key:
            #         if z in shared_slices-1:
            #             nodes_to_left_of_boundary[key][con_comp_label].append(tuple(coord))
            #         else:
            #             nodes_to_right_of_boundary[key][(tuple(coord))] = con_comp_label
            

    ################################################################################################################
    # Check the 26 boxes surrounding each node that lives on the left side of chunking indices
    # Update full neighbors dictionary
    # Create dictionary of conn components that need to merge across slice index.
    #   Because they were previously thought of as two separate components but now we need to update all of our records
    #   to indicate they are one
    ################################################################################################################

    movement_vectors = ([p for p in itertools.product([0,1,-1], repeat=3)])
    full_merge_dict = defaultdict(set)
    merging_ccs = defaultdict(set)

    for slice_locations in shared_slices:
        # print(slice_locations)
        left_side = slice_locations-1
        right_side = slice_locations
        
        #Iterate through Left Conn Components that have nodes on the boundary
        #Find nodes on the other side and their corresponding CC label indicating a need to merge
        
        for cc_label in nodes_to_left_of_boundary[left_side].keys():       
            # print(cc_label)
            
            cc_coords_to_check = nodes_to_left_of_boundary[left_side][cc_label]
            for left_node in cc_coords_to_check:
                for vect in movement_vectors:
                    node_to_check_on_other_side = tuple(list(map(add,tuple(left_node),vect)))
                    if node_to_check_on_other_side in nodes_to_right_of_boundary[right_side]:
                        right_cc = nodes_to_right_of_boundary[right_side][node_to_check_on_other_side]                  
                        
                        #Update Neighbors Dictionary
                        # print('IM ADDING {} to {} Neighbor Dict'.format(node_to_check_on_other_side,left_node))
                        full_neighbors_dict[cc_label][left_node].add(node_to_check_on_other_side)                    
                        full_neighbors_dict[right_cc][node_to_check_on_other_side].add(left_node)
                        
                        merging_ccs[cc_label].add(right_cc)
                        # print(merging_ccs)
    
    full_merge_dict = consolidate_conn_components(merging_ccs) 
                                                                   
    ################################################################################################################
    #Merge Connected Components Across Chunk Slices
    ################################################################################################################

    #merging these values in full neighbors dict
    for keeping_cc,merging_cc in full_merge_dict.items():
        for merge_cc in merging_cc:           
            #pdate full neighbors dict
            full_neighbors_dict[keeping_cc].update(full_neighbors_dict[merge_cc])
            
            del full_neighbors_dict[merge_cc]

        
    ################################################################################################################
    #Loads soma segmentation coordinates. Compress to x-y plane, run connected components to determine number of somas. Picks soma closest to center of image 
    ################################################################################################################


    no_soma = False
    p = os.path.join(specimen_dir,"MAX_Segmentation_ch1.tif")
    if not os.path.exists(p):
        l = os.path.join(specimen_dir,"MAX_Left_Segmentation_ch1.tif")
        r = os.path.join(specimen_dir,"MAX_Right_Segmentation_ch1.tif")

        if all([os.path.exists(seg_path) for seg_path in [l,r]]):
            l_mip = tif.imread(l)
            r_mip = tif.imread(r)
            ch1_mip = np.hstack((l_mip,r_mip))

        else:
            #make a fake centroid that we'll delete later
            print("{} Is Missing max segmentation ch1 tif. Creating placeholder soma".format(sp_id))
            no_soma = True
            centroid = (0,0,0)
            soma_connection_threshold = 0
    else:
        ch1_mip = tif.imread(p)

    if no_soma == False:

        able_to_id_soma_from_mip = True
        original_ch1_mip = ch1_mip.copy()

        xs_63x,ys_63x = None, None # get_63x_soma_coords(specimen_id) this was an AIBS internal function where we would query a database to get the image coordinate X and Y location for the soma node
        bub = 300
        # If 63x soma coords are available GREAT, we will use those and define soma location that way
        if (xs_63x!= None) and (ys_63x != None):
            avg_x = np.mean(xs_63x)
            avg_y = np.mean(ys_63x)
            # no soma is going to be bigger than 600 x 600 pixels, right?
            x0,x1 = int(avg_x-bub), int(avg_x+bub)
            y0,y1 = int(avg_y-bub), int(avg_y+bub)
            #ROI max
            mip_max = ch1_mip[y0:y1,x0:x1].max()
            #zero out non ROI
            ch1_mip[0:y0:,0:x0] = 0
            ch1_mip[y1:,x1:] = 0

        else:
            mip_max = ch1_mip.max()

        #create a minimum signal value for our ch1 segmentation
        if mip_max < 100:
            cutoff = math.ceil(mip_max*0.15)
        else:
            cutoff = math.ceil(mip_max*0.3)

        print("Mip max = {}, cutoff = {}".format(mip_max,cutoff))
        ch1_mip[ch1_mip<cutoff] = 0

        struct = [[1,1,1],
        [1,1,1],
        [1,1,1]]
        conn_comps, num_comps = label(ch1_mip,structure=struct)

        if (xs_63x!= None) and (ys_63x != None):
            # use the connected component label where our database tells us the soma is
            soma_cc_label = conn_comps[int(avg_y),int(avg_x)]
            
        else:
            # otherwise use the center most soma. This is an assumption that the cell of interests soma is closest to the images center
            img_shape = ch1_mip.shape
            # need to find the center most connected components label
            cc_count = 0
            list_of_cc_labels = []
            for i in range(num_comps+1):
                if i != 0:
                    index = np.where(conn_comps == i)
                    if len(index[0]) > minimum_soma_area_pixels:
                        cc_count+=1
                        list_of_cc_labels.append(i)

            # #Threshold was too aggressive
            # if list_of_cc_labels == []:
            #     for i in range(num_comps+1):
            #         if i != 0:
            #             index = np.where(conn_comps == i)
            #             cc_count+=1
            #             list_of_cc_labels.append(i)

            #if its still empty just default to 0
            if list_of_cc_labels == []:
                able_to_id_soma_from_mip = False
                print("No soma found in Ch1 Max Intensity Image. Defaulting soma to center of image")
                soma_cc_label = 0
                no_soma = True
                centroid = (0, 0, 0)
                soma_connection_threshold = 0


            print('There are {} somas in {}'.format(cc_count,sp_id))
            
            #take the connected coponent that's center is closest to the center of the image
            middle_of_image_x = int(img_shape[1]/2)
            middle_of_image_y = int(img_shape[0]/2)
            closest_distance = np.inf
            for cc_ind in list_of_cc_labels:
                index = np.where(conn_comps == cc_ind)
                conn_comp_xs = index[1]
                conn_comp_ys = index[0]
                cc_mean_x = np.mean(conn_comp_xs)
                cc_mean_y = np.mean(conn_comp_ys)

                this_cc_dist_to_center = distance.euclidean((cc_mean_y,cc_mean_x),
                                                            (middle_of_image_y,middle_of_image_x))

                if this_cc_dist_to_center < closest_distance:
                    soma_cc_label = cc_ind
                    closest_distance = this_cc_dist_to_center


        if able_to_id_soma_from_mip:
            chosen_ys,chosen_xs = np.where(conn_comps==soma_cc_label)

            #binarize mip so only chosen soma cc location remains
            conn_comps[conn_comps!=soma_cc_label] = 0
            conn_comps[conn_comps==soma_cc_label] = 1

            #check that this chosen connected component isn't a massive ball of soma fill
            max_dx_soma_cc = max(chosen_xs)-min(chosen_xs)
            max_dy_soma_cc = max(chosen_ys)-min(chosen_ys)

            soma_size_thresh = 2*bub - 20
            erosion_stopping_thresh = 50
            #here we will erode the soma conn comp until it fits into our acceptable size
            if (max_dx_soma_cc > soma_size_thresh) or (max_dy_soma_cc>soma_size_thresh):
                print("Going to erode the soma connected component a bit because it exceeds our size threshold")
                eroded_conn_comp = conn_comps.copy()
                temp_xs,temp_ys = np.where(conn_comps==1)
                temp_dx = max(temp_xs)-min(temp_xs)
                temp_dy = max(temp_ys)-min(temp_ys)
                stopping_condition = False
                while stopping_condition == False:
                    eroded_conn_comp = morphology.binary_erosion(eroded_conn_comp).astype(eroded_conn_comp.dtype)
                    temp_xs,temp_ys = np.where(eroded_conn_comp==1)
                    temp_dx = max(temp_xs)-min(temp_xs)
                    temp_dy = max(temp_ys)-min(temp_ys)

                    if (temp_dx < soma_size_thresh) and (temp_dy<soma_size_thresh):
                        print("   temp dx {} and dy {} is less than our threshold {}".format(temp_dx,temp_dy,soma_size_thresh))
                        stopping_condition = True
                    elif (temp_dx < erosion_stopping_thresh) or (temp_dy<erosion_stopping_thresh):
                        #we dont want to completely erode away so we will stop
                        print("   Woah there. One of our dimensions has been eroded to our erosion threhsold")
                        print(temp_dx,temp_dy)
                        stopping_condition = True


                conn_comps = eroded_conn_comp.copy()
                chosen_ys,chosen_xs = np.where(conn_comps==1)

            min_y,max_y = int(min(chosen_ys)),int(max(chosen_ys))
            min_x,max_x = int(min(chosen_xs)),int(max(chosen_xs))

            #Now time to figure out z coordinates
            yz_mip = os.path.join(specimen_dir,"MAX_yz_Segmentation_ch1.tif")
            if not os.path.exists(yz_mip):

                l_pth = os.path.join(specimen_dir,"MAX_yz_Left_Segmentation_ch1.tif")
                r_pth = os.path.join(specimen_dir,"MAX_yz_Right_Segmentation_ch1.tif")
                o_pth = os.path.join(specimen_dir,"MAX_yz_Segmentation_ch1.tif")

                l_yz_mip = tif.imread(l_pth)
                r_yz_mip = tif.imread(r_pth)

                new_stack = np.zeros((l_yz_mip.shape[0],l_yz_mip.shape[1],2))
                new_stack[:,:,0] = l_yz_mip
                new_stack[:,:,1] = r_yz_mip

                yz_mip = new_stack.max(axis=2)
                tif.imwrite(o_pth,yz_mip)

            else:
                yz_mip = tif.imread(yz_mip)

            #zero out any irrelevant y values
            yz_mip[:,0:min_y] = 0
            yz_mip[:,max_y:0] = 0

            #thresholding
            yz_mip_roi_max = yz_mip.max()
            if yz_mip_roi_max < 100:
                yz_cutoff = int(yz_mip_roi_max*0.15)
            else:
                yz_cutoff = int(yz_mip_roi_max*0.3)

            #iterate and choose largest connected component in this y-column
            yz_mip[yz_mip<yz_cutoff] = 0
            yz_conn_comps, yz_num_comps = label(yz_mip)
            biggest_component_size = 0
            for i in range(yz_num_comps+1):
                if i != 0:
                    index = np.where(yz_conn_comps == i)
                    size_comp = len(index[0])
                    if size_comp > biggest_component_size:
                        biggest_component_size = size_comp
                        yz_conn_comp_label = i

            yz_conn_comps[yz_conn_comps!=yz_conn_comp_label] = 0
            yz_conn_comps[yz_conn_comps==yz_conn_comp_label] = 1

            # yz_eroded_soma = morphology.binary_erosion(yz_conn_comps).astype(yz_conn_comps.dtype)
            # yz_perimeter = yz_conn_comps - yz_eroded_soma
            # perimeter_zs,perimeter_ys_yz = np.where(yz_perimeter!=0)

            chosen_zs,_ = np.where(yz_conn_comps == 1)
            centroid = (np.mean(chosen_xs),np.mean(chosen_ys),np.mean(chosen_zs))
            mean_diameter = ((max_y-min_y)+(max_x-min_x))/2
            mean_radius = mean_diameter/2

            soma_connection_threshold = mean_radius*2.25

            #make helper plot
            fig,axe=plt.subplots(1,2)
            axe[0].imshow(original_ch1_mip)
            axe[0].set_title("Ch1 XY MIP")
            axe[1].imshow(conn_comps)
            axe[1].set_title("Chosen Conn Comp + centroid")
            axe[1].scatter(centroid[0],centroid[1],c='r',s=5)
            for i in [0,1]:
                axe[i].set_xlim(min_x-bub,max_x+bub)
                axe[i].set_ylim(min_y-bub,max_y+bub)
            ofile = os.path.join(specimen_dir,"SomaSelectionImg.png")
            fig.set_size_inches(8,3.5)
            fig.savefig(ofile,dpi=300,bbox_inches='tight')
            plt.clf()


    print("Soma Centroid is at {}. Soma connection radius = {}".format(centroid,soma_connection_threshold))


    ################################################################################################################
    # Find Starting Node for each conn component. Check if its within 50 pixels of soma. Assign parent accordingly
    ################################################################################################################
    #The somas parent is always -1
    parent_dict = {}
    parent_dict[centroid] = -1

    for conn_comp in full_neighbors_dict.keys():
        # print('at conn_comp {}'.format(conn_comp))
        neighbors_dict = full_neighbors_dict[conn_comp]
        #find "leaf nodes"
        if len(full_neighbors_dict[conn_comp]) > 2:          
            leaf_nodes = [x for x in neighbors_dict.keys() if len(neighbors_dict[x]) == 1]

            # There is no leaf node to start (somehow there is a loop), so we will break it
            if leaf_nodes == []:
                # find node closest to soma
                dist_dict = {}
                for coord in full_neighbors_dict[conn_comp].keys():
                    dist_to_soma = euclidean(centroid,coord)
                    dist_dict[coord] = dist_to_soma
                start_node = min(dist_dict, key=dist_dict.get)
                while len(full_neighbors_dict[conn_comp][start_node])>1:###REMOVE CYCLE
                    print('removing cycle')
                    removed = full_neighbors_dict[conn_comp][start_node].pop()
                    full_neighbors_dict[conn_comp][removed].discard(start_node)

                # Check how far it is from soma cloud
                dist = distance.euclidean(centroid,start_node)
                
                # print('Distance = {}'.format(dist))
                if dist < soma_connection_threshold:
                    print('assigning soma centroid as the start node')
                    start_parent = centroid
                else:
                    start_parent = 0
                
                assign_parent_child_relation(start_node,start_parent,parent_dict,neighbors_dict)

            # At least one leaf node exists
            else:
                dist_dict = {}
                for coord in leaf_nodes:
                    dist_to_soma = euclidean(centroid,coord)
                    dist_dict[coord] = dist_to_soma
                #start node is the leaf node that is closest to the soma
                start_node = min(dist_dict, key=dist_dict.get)

                # Check how far it is from soma cloud
                dist = distance.euclidean(centroid,start_node)
                
                # print('Distance = {}'.format(dist))
                if dist < soma_connection_threshold:
                    # print('assigning soma centroid as the start node')
                    start_parent = centroid
                else:
                    # This will be a disconnected segment
                    start_parent = 0
                
                assign_parent_child_relation(start_node,start_parent,parent_dict,neighbors_dict)

    # In case with dummy centroid remove centroid from parent dict and centroid
    if no_soma == True:
        parent_dict.pop(centroid)
        for k,v in parent_dict.items():
            if v == centroid:
                parent_dict[k] = -1 

    # number each node for sake of swc
    ct=0
    big_node_dict = {}
    for j in parent_dict.keys():
        ct+=1
        big_node_dict[tuple(j)] = ct
        
    # Load node type labels
    skeleton_labeled = pd.read_csv(skeleton_labels_file)
    skeleton_coord_labels_dict = {}
    for n in skeleton_labeled.index: 
        skeleton_coord_labels_dict[(skeleton_labeled.loc[n].values[0],skeleton_labeled.loc[n].values[1],skeleton_labeled.loc[n].values[2])] = skeleton_labeled.loc[n].values[3] 

    # Make swc list for swc file writing
    swc_list = []
    for k,v in parent_dict.items():
        # id,type,x,y,z,r,pid
        if v == 0:
            parent = -1
            node_type = skeleton_coord_labels_dict[k]
            radius = 1
        elif v == -1:
            parent = -1
            node_type = 1
            radius = mean_radius
        else:
            parent = big_node_dict[v]
            node_type = skeleton_coord_labels_dict[k]
            radius = 1

        swc_line = [big_node_dict[k]] + [node_type] + list(k) + [radius] + [parent]

        swc_list.append(swc_line)
        
    # Write Swc file
    swc_path = os.path.join(specimen_dir,'{}_raw_autotrace.swc'.format(sp_id))
    with open(swc_path,'w') as f:
        f.write('# id,type,x,y,z,r,pid')
        f.write('\n')
        for sublist in swc_list:
            for val in sublist:
                f.write(str(val))
                f.write(' ')
            f.write('\n')

    print('finished writing swc for specimen {}'.format(sp_id))

    # Cleanup
    if remove_intermediate_files:

        directories_to_remove = ["Chunks_of_32","Chunks_of_32_Left","Chunks_of_32_Right",
                                "Segmentation","Left_Segmentation","Right_Segmentation",
                                "Skeleton","Left_Skeleton","Right_Skeleton",
                                "Single_Tif_Images","Single_Tif_Images_Left","Single_Tif_Images_Right"]
        print("Cleaning Up:")
        for dir_name in directories_to_remove:
            full_dir_name = os.path.join(specimen_dir,dir_name)
            if os.path.exists(full_dir_name):
                print(full_dir_name)
                shutil.rmtree(full_dir_name)
    #
    #

def main(specimen_id,specimen_dir,remove_intermediate_files, max_stack_size, minimum_soma_area_pixels, soma_connection_threshold, **kwargs):

    skeleton_to_swc_parallel(specimen_id,specimen_dir,remove_intermediate_files,max_stack_size,minimum_soma_area_pixels,soma_connection_threshold)



if __name__ == "__main__":
	module = ags.ArgSchemaParser(schema_type=LayerSchema)
	main(**module.args)
