function [T_all, D_all] = grid2im(dir_name, im_size)
% This function takes a directory, obtains an index for each patch (x_i, y_i, z_i)
% from and returns the position in the
% grid.
 
D_all = false(im_size);
T_all = false(im_size);
mat_files = dir([dir_name, '*.mat']);
mat_files = {mat_files(:).name}';
 
for i = 1:numel(mat_files)
    curr_mat_file = [dir_name, mat_files{i}];
    patch_T = load(curr_mat_file,'T'); % T
    patch_T = patch_T.T;
    %patch_size = size(patch_T);
    patch_size = [400,400,100];
    
    patch_D = load(curr_mat_file,'D'); % T
    patch_D = patch_D.D;
    
    split_fn = regexp(mat_files{i}, '_', 'split');
    x_i = str2double(split_fn{2});
    y_i = str2double(split_fn{3});
    split_fn2 = regexp(split_fn{4}, '.mat', 'split');
    z_i = str2double(split_fn2{1});
    
    %These are co-ordinates in IM space
    px_min = (x_i-1)*patch_size(1)+1;
    py_min = (y_i-1)*patch_size(2)+1;
    pz_min = (z_i-1)*patch_size(3)+1;
    
    px_max = min((x_i)*patch_size(1),im_size(1));
    py_max = min((y_i)*patch_size(2),im_size(2));
    pz_max = min((z_i)*patch_size(3),im_size(3));
   
    
    patch_T = bsxfun(@plus, patch_T, 1);
    patch_T(isinf(patch_T))=0;
    patch_T = patch_T>0;
    
    patch_D = bsxfun(@plus, patch_D, 1);
    patch_D(isinf(patch_D))=0;
    patch_D = patch_D>0;

    T_all(px_min:px_max,py_min:py_max,pz_min:pz_max) = patch_T(1:(px_max-px_min+1),1:(py_max-py_min+1),1:(pz_max-pz_min+1));
    D_all(px_min:px_max,py_min:py_max,pz_min:pz_max) = patch_D(1:(px_max-px_min+1),1:(py_max-py_min+1),1:(pz_max-pz_min+1));
end
 
end
 
