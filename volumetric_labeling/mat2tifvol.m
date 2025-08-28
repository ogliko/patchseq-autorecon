function [] = mat2tifvol(Mat,fname)
%The third dimension is saved as different planes.

imwrite(Mat(:, :, 1), fname);
for k = 2:size(Mat, 3)
%     imwrite(Mat(:, :, k), fname, 'WriteMode', 'append', 'Compression', 'none');
    imwrite(Mat(:, :, k), fname, 'WriteMode', 'append');
end