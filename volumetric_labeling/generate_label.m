% This script is used to generate a volumetric label from manual neuron 
% trace and image stack using topology preserving fast marching algorithm.
% 'Single_Tif_Images': Folder containing an image stack with individual
% TIFF files per z-plane.
% 'trace.swc': Manual trace of neuron in image coordinates.
% Authors: Olga Gliko, Rohan Gala, Uygar Sumbul

clear
close all

root_dir = fullfile('/data/');
im_dir = fullfile(root_dir,'Single_Tif_Images/');
swc_file = fullfile(root_dir, 'trace.swc');
disp(root_dir)
disp(im_dir)
disp(swc_file)
save_dir = fullfile(root_dir,'preprocess/');
mkdir(save_dir);

% Set soma threshold
soma_threshold = 140; % range 140-220

% Generate label
MaxDist = 6;
patch_size=[400,400,100];

[T_all, ~, ~, ~, r] = FM_patch(im_dir, swc_file, save_dir, MaxDist, patch_size, soma_threshold);
T_all = uint8(T_all);

trace = loadswc(swc_file);
[y, x, z] = ind2sub(size(T_all), find(T_all));
foreground = [x, y, z];
for i = 1:length(z)
    dist = pdist2(foreground(i,:), trace(:,3:5), 'euclidean');
    [~, min_dist_idx] = min(dist);
    node_type = trace(min_dist_idx, 2);
    % label apical dendrite as dendrite node_type=3
    if node_type == 4
        node_type = 3;
    end
    T_all(y(i), x(i), z(i)) = node_type;
end    

load(fullfile(save_dir, 'soma.mat'));
idx = find(thresh_soma > 0);
soma_patch = T_all(sx_min:sx_max,sy_min:sy_max,sz_min:sz_max);
soma_patch(idx) = 1;
T_all(sx_min:sx_max,sy_min:sy_max,sz_min:sz_max) = soma_patch;

labels_dir = fullfile(save_dir,'labels/');
mkdir(labels_dir);
mat2tifslice(T_all,labels_dir);
