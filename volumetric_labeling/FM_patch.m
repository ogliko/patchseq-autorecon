function [T_all, D_all, thresh_soma, soma_patch, r] = FM_patch(im_dir, swc_file, results_dir, MaxDist, patch_size, threshold)
patches_dir = fullfile(results_dir, 'patches/'); % OG changed
mkdir(patches_dir);

% Read trace
[AMlbl, r, R, t, ~] = swc2AM(swc_file);
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
start_im_offset = str2double(im_files{1}(end-7:end-4))-1;

% Retrieve soma patch, thresholded patch, and r
[thresh_soma,soma_patch,r,sx_min,sx_max,sy_min,sy_max,sz_min,sz_max] = retrieve_soma(r,R,t,MaxDist,im_fileid,start_im_offset,threshold); % OG changed

% Run fast marching on each patch
ceil_coor = ceil(im_size./patch_size);
n_patches = ceil_coor;
for z_i = 1:n_patches(3)
    for x_i = 1:n_patches(1)
        for y_i = 1:n_patches(2)
            px_min = (x_i-1)*patch_size(1)+1;
            py_min = (y_i-1)*patch_size(2)+1;
            pz_min = (z_i-1)*patch_size(3)+1;
            
            px_max = min((x_i)*patch_size(1),im_size(1));
            py_max = min((y_i)*patch_size(2),im_size(2));
            pz_max = min((z_i)*patch_size(3),im_size(3));
            
            r_ind = round(r(:,1))>=px_min & round(r(:,1))<=px_max & ...
                round(r(:,2))>=py_min & round(r(:,2))<=py_max & ...
                round(r(:,3))>=pz_min & round(r(:,3))<=pz_max;
            
            r_patch = bsxfun(@minus,r(r_ind,:),[px_min,py_min,pz_min])+1;
            is_trace_present = ~isempty(r_patch);
            
            if is_trace_present
                [patch] = fetch_patch(px_min,px_max,py_min,py_max,pz_min,pz_max,im_fileid,start_im_offset);
                [nonsimple,KT,D,T]=FastMarchingTube(patch,[r_patch(:,1),r_patch(:,2),r_patch(:,3)],MaxDist,[1,1,0.3]);
                savefile = fullfile(patches_dir, ['FM_', num2str(x_i), '_', num2str(y_i), '_', num2str(z_i),'.mat']); % OG changed
                save(savefile, 'nonsimple','KT','D','T','patch','r_patch');
            end
        end
    end
end

save([results_dir, 'soma.mat'],'soma_patch','thresh_soma','sx_min','sx_max','sy_min','sy_max','sz_min','sz_max');
[T_all, D_all] = grid2im(patches_dir, im_size); % OG changed
end
