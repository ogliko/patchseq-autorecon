function [AMlbl,r,R,t,SWC]=swc2AM(pth_or_swc)
%This function converts swc format to .mat, AMlbl r R format.

if exist(pth_or_swc,'file')
    temp=dir(pth_or_swc);
    temp=temp.name;
    ind=find(temp=='.',1,'last');
    if ~strcmp(temp(ind+1:end),'swc') && ~strcmp(temp(ind+1:end),'SWC')
        AMlbl=[];
        r=[];
        R=[];
        disp('Incorrect file format')
        return
    end
    
    fileID=fopen(pth_or_swc);
    SWC=textscan(fileID,'%f %f %f %f %f %f %f','commentStyle','#');
    fclose all;
    SWC=cell2mat(SWC);
    
elseif size(pth_or_swc,2)==7
    SWC=pth_or_swc;
else
    AMlbl=[];
    r=[];
    R=[];
    disp('Incorrect path or SWC file format')
    return
end

N=size(SWC,1);
% r ranges from 0.5 to sizeIm+0.5 in Matlab and [0 sizeIm] in Java and SWC
r=[SWC(:,4),SWC(:,3),SWC(:,5)]+0.5;
R=SWC(:,6);
t=SWC(:,2);
AM=sparse(N,N);
roots=(SWC(:,7)==-1);

% changed to match swc indices
ind=sub2ind(size(AM),SWC(~roots,1),SWC(~roots,7));
AM(ind)=1;
AMlbl=LabelTreesAM(AM);
