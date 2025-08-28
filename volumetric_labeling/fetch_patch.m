function [patch] = fetch_patch(px_min,px_max,py_min,py_max,pz_min,pz_max,fileid,start_im_offset)
z_range = pz_min:pz_max;
num_images = numel(z_range);
m_im = numel(px_min:px_max);
n_im = numel(py_min:py_max);
patch = zeros(n_im,m_im,num_images);
for i=1:num_images
    file_tif = [fileid, num2str(z_range(i)+start_im_offset, '%04d'),'.tif'];
    file_lib = Tiff(file_tif,'r');
    rps = file_lib.getTag('RowsPerStrip');
    rps = min(rps,m_im);
    row_count = 1;
    for row=px_min:rps:px_max
        row_inds = row_count:min(m_im,row_count+rps-1);
        strip_num = file_lib.computeStrip(row);
        curr_strip = file_lib.readEncodedStrip(strip_num);
        row_ind_size = size(row_inds);
        patch(row_inds,1:n_im,i) = curr_strip(row_ind_size(1):row_ind_size(2),py_min:py_max);
        row_count=row_count+rps;
    end
end
end

