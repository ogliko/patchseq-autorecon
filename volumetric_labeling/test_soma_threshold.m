%%
% This script is used to find optimal soma intensity threshold.
% 'Single_Tif_Images': Folder containing an image stack with individual
% TIFF files per z-plane.
% 'trace.swc': Manual trace of neuron in image coordinates.
% Authors: Olga Gliko, Rohan Gala, Uygar Sumbul.

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
%%
% Set soma threshold
threshold = 145; % range 140-220

% Read trace
[AMlbl, r, R, t, ~] = swc2AM(swc_file); % r - swc y,x,z; R - swc r; t - swc type
[~, r, R] = AdjustPPM(AMlbl, r, R, .5);

% File names for stack
im_files = dir([im_dir, '*.tif']);
im_files = {im_files(:).name}';

% Get tif file parameters
im_1 = Tiff([im_dir, im_files{1}]);
im_width = im_1.getTag('ImageWidth');
im_height = im_1.getTag('ImageLength');
im_size = [im_height, im_width, numel(im_files)];
im_fileid = [im_dir,im_files{1}(1:end-8)];
start_im_offset = str2double(im_files{1}(end-4))-1;

MaxDist = 6;  

% Retrieve soma patch, thresholded patch, and r
[thresh_soma,soma_patch,~,~,~,~,~,~,~] = retrieve_soma(r,R,t,MaxDist,im_fileid,start_im_offset,threshold);

fprintf('max_soma_patch: %d\n', max(soma_patch(:)))
figure('Name','soma patch');
clf;
imshow(max(soma_patch,[],3),[]);

fprintf('max_thresh_soma: %d\n', max(thresh_soma(:)))
figure('Name','thresh_soma');
clf;
imshow(max(thresh_soma,[],3),[]);
%%
% Test another soma threshold
threshold = 140;
[thresh_soma_new] = test_new_threshold(soma_patch, threshold);

fprintf('max_thresh_soma_new: %d\n', max(thresh_soma_new(:)))
figure('Name', 'thresh soma new');
clf;
imshow(max(thresh_soma_new,[],3),[]);

%%
function [thresh_soma] = test_new_threshold(soma_patch, threshold)
SE = strel('sphere',5);
eroded_soma = imerode(soma_patch, SE);
eroded_soma = imerode(eroded_soma, SE);
eroded_soma = imerode(eroded_soma, SE);
dilated_soma = imdilate(eroded_soma, SE);
dilated_soma = imdilate(dilated_soma, SE);
thresh_soma = dilated_soma;
thresh_soma(dilated_soma<threshold) = 0;
thresh_soma(thresh_soma>0) = 1;
end