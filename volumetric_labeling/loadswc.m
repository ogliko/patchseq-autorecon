function [swc] = loadswc(filepath)
% Loads a swc file as an N x 7 matrix

fileID = fopen(filepath, 'r');
swc=textscan(fileID,'%f %f %f %f %f %f %f','commentStyle','#');
fclose all;
swc=cell2mat(swc);
end