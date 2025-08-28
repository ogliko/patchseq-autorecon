function [] = mat2tifslice(IM,dirname)
%IM is a matrix with image data
%dirname must end with /
for i=1:size(IM,3)
%     imwrite(IM(:,:,i),[dirname,num2str(i),'.tif'],'Compression','none')
    imwrite(IM(:,:,i),fullfile(dirname, sprintf('%03d.tif', i)),'Compression','none');
    %disp(i)
end
end