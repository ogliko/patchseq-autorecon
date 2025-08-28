% This function imports images into MatLab. RGB images are converted to
% grayscale. Data format of Orig is preserved (uint8, uint16, etc).

function [Orig,sizeOrig,classOrig]=ImportStackJ(pth,file_list)

Orig=[];
sizeOrig=[];
classOrig=[];

N=length(file_list);
info = imfinfo([pth,file_list{1}]);
Npl=length(info);

if N==0
    disp('There are no images which pass the file selection criteria.')
    return
    
elseif N==1 && Npl>1 %import a virtual stack (tif or LSM)
    temp = imread([pth,file_list{1}],'Index',1);
    classOrig=class(temp);
    formatOrig=size(temp,3);
    sizeOrig=fix([size(temp,1),size(temp,2),Npl]);
    Orig=zeros(sizeOrig,classOrig);
    for i=1:Npl
        temp = imread([pth,file_list{1}],'Index',i);
        if formatOrig==3
            temp=rgb2gray(temp);
        end
        Orig(:,:,i) = temp;
    end
    
    
elseif N>1 || (N==1 && Npl==1) %import a set of regular images
    
    if Npl>1
        disp('Unable to load multiple virtual stacks.')
        return
    end
    
    temp = imread([pth,file_list{1}]);
    for i=2:N
        info = imfinfo([pth,file_list{i}]);
        if length(info)>1
            disp('Unable to load multiple virtual stacks.')
            return
        end
        if info.Height~=size(temp,1) || info.Width~=size(temp,2)
            disp('Unable to load. Images have different dimensions.')
            return
        end
    end
    
    classOrig=class(temp);
    formatOrig=size(temp,3);
    sizeOrig=fix([size(temp,1),size(temp,2),N]);
    Orig=zeros(sizeOrig,classOrig);    
    for i=1:N
        temp = imread([pth,file_list{i}]);
        if formatOrig==3
            temp=rgb2gray(temp);
        end
        Orig(:,:,i) = temp;
    end
    
end
