% This function the Eikonal equation by using the Fast Marching algorithm
% of Sethian. T and D are the arival time and distance maps.
% Max_Known_Dist is the distance at which re-initialization is performed
% SVr contains positions of the seeds
% unisotropy is the wave speed unisotropy in a uniform intensity image
% Output is Logical KT==1;
%  13    13     1
function [nonsimple,KT,D,T]=FastMarchingTube(IM,SVr,Max_Known_Dist,unisotropy)
output=true;
pad=2;
Max_Known_Time=inf;

unisotropy(unisotropy<0)=0;
if sum(unisotropy)==0
    unisotropy=[1,1,1];
end
unisotropy=unisotropy./sum(unisotropy.^2)^0.5;
h2=[unisotropy(1),unisotropy(1),unisotropy(2),unisotropy(2),unisotropy(3),unisotropy(3)].^2;

%Orig=ones(sizeIM)*1;
Orig = IM;
sizeOrig=size(Orig);
if length(sizeOrig)==2
    sizeOrig=[sizeOrig,1];
end

Im=zeros(sizeOrig+2*pad);
Im(1+pad:end-pad,1+pad:end-pad,1+pad:end-pad)=Orig;
clear Orig
sizeIm=sizeOrig+2*pad;

SVr=round(SVr);%Matlab IM pixel goes from 0.5 to 1.5 etc.
SVr=SVr+pad;
StartVoxel=sub2ind(sizeIm,SVr(:,1),SVr(:,2),SVr(:,3));
StartVoxel=unique(StartVoxel);

N6_ind=[-1;+1;-sizeIm(1);+sizeIm(1);-sizeIm(1)*sizeIm(2);+sizeIm(1)*sizeIm(2)];
S=false(sizeIm); 
nsimp = 1;
%KTcopy=false(sizeIm); 
was_nonsimple=nan(10^5,1);
T=inf(sizeIm,'single'); T(StartVoxel)=0;
D=inf(sizeIm,'single'); D(StartVoxel)=0;
KT=zeros(sizeIm,'uint8'); KT(StartVoxel)=1; % Known=1, Trial=2, Unvisited=0
KTcopy=KT;
KnownPoints=StartVoxel;

NHood=[KnownPoints+N6_ind(1);KnownPoints+N6_ind(2);KnownPoints+N6_ind(3); ...
    KnownPoints+N6_ind(4);KnownPoints+N6_ind(5);KnownPoints+N6_ind(6)];
NHood(KT(NHood)==1)=[];
NHood(Im(NHood)==0)=[]; %For padding
NHood=unique(NHood);

TrialPoints=NHood(~isinf(T(NHood)));
T_TrialPoints=T(TrialPoints);

stop_cond=true;
if isempty(NHood)
    stop_cond=false;
    exit_flag='end';
end

NewKnownPoint=[];
iter=0;
while stop_cond
    iter=iter+1;
    if mod(iter,1000)==0 && output
        display(['Current distance: ', num2str(D(NewKnownPoint)),' :: Current time: ', num2str(T(NewKnownPoint))]);
    end
    
    if ~isempty(NHood)
        ind=[NHood+N6_ind(1),NHood+N6_ind(2),NHood+N6_ind(3),NHood+N6_ind(4),NHood+N6_ind(5),NHood+N6_ind(6)];
        KT_1=(KT(ind)~=1);
        
        Tpm_xyz=T(ind);
        Tpm_xyz(KT_1)=inf;
        [Tpm_xyz,IX]=sort(Tpm_xyz,2);
        
        Dpm_xyz=D(ind);
        Dpm_xyz(KT_1)=inf;
        Dpm_xyz=sort(Dpm_xyz,2);
        
        
        %Calculate arrival time, T, and distance, D, for the trial
        %points------------------------------------------------------------
        H=h2(IX);
        H_cum=cumsum(H,2);
        ct=1./Im(NHood).^2;
        Tpm_xyz_cum=cumsum(Tpm_xyz.*H,2);
        Tpm_xyz2_cum=cumsum(Tpm_xyz.^2.*H,2);
        nt=sum(((Tpm_xyz2_cum-Tpm_xyz_cum.^2./H_cum)<=ct*ones(1,size(Tpm_xyz,2))),2);
        if sum(h2==0)>0
            nt(nt==0)=1;
        end
        ind_nt=(1:size(Tpm_xyz,1))'+(nt-1).*size(Tpm_xyz,1);
        ntH=H_cum(ind_nt);
        temp=Tpm_xyz_cum(ind_nt);
        T(NHood)=(temp+(ct.*ntH-(Tpm_xyz2_cum(ind_nt).*ntH-temp.^2)).^0.5)./ntH;
        
        N=ones(size(Tpm_xyz,1),1)*(1:size(Tpm_xyz,2));
        Dpm_xyz_cum=cumsum(Dpm_xyz,2);
        Dpm_xyz2_cum=cumsum(Dpm_xyz.^2,2);
        nd=sum(((Dpm_xyz2_cum-Dpm_xyz_cum.^2./N)<=ones(size(Dpm_xyz))),2);
        ind_nd=(1:size(Dpm_xyz,1))'+(nd-1).*size(Dpm_xyz,1);
        temp=Dpm_xyz_cum(ind_nd);
        D(NHood)=(temp+(nd-(Dpm_xyz2_cum(ind_nd).*nd-temp.^2)).^0.5)./nd;
        %------------------------------------------------------------------
        
        keep=NHood(KT(NHood)==0 & ~isinf(T(NHood)));
        KT(keep)=2;
        TrialPoints=[TrialPoints;keep];
        T_TrialPoints=[T_TrialPoints;T(keep)];
    end
    
    [min_T,min_ind]=min(T_TrialPoints);
    NewKnownPoint=TrialPoints(min_ind);
    
    %Check conditions for newest point added to the list of known points
    if ~isempty(NewKnownPoint) && ~isinf(min_T)
        KT(NewKnownPoint)=1;
        KTcopy(NewKnownPoint)=1;
        T_TrialPoints(min_ind)=inf;
        NHood=NewKnownPoint+N6_ind;
        KnownPoints=[KnownPoints;NewKnownPoint];
        
        NHood(KT(NHood)==1)=[];
        NHood(Im(NHood)==0)=[]; %Remove points in padding
        
        [nx,ny,nz] = ind2sub(size(S),NewKnownPoint);
        
        %Debug plots
        %figure(100);
        %plot(ny,nx,'.b','MarkerSize',10);hold on;drawnow
        %KTcopy(was_nonsimple(1:nsimp-1))=0;
        if simple3d(KTcopy(nx-1:nx+1,ny-1:ny+1,nz-1:nz+1)==1,6)==0
            S(NewKnownPoint)=1;
            was_nonsimple(nsimp)=NewKnownPoint;
            nsimp = nsimp+1;
            KTcopy(NewKnownPoint)=0;
            %plot(ny,nx,'.c','MarkerSize',20);hold on;drawnow
        end
        
        if T(NewKnownPoint)>=Max_Known_Time
            stop_cond=false;
            exit_flag='time'; % maximum time reached
        end
        if D(NewKnownPoint)>=Max_Known_Dist
            stop_cond=false;
            exit_flag='dist'; % maximum distance reached
        end
    else
        stop_cond=false;
        exit_flag='end'; % No place to propagate.
    end
end

[nx,ny,nz] = ind2sub(size(KT),was_nonsimple(1:nsimp-1));
nonsimple = [nx-pad,ny-pad];

KT=(KT==1);
KT=KT(1+pad:end-pad,1+pad:end-pad,1+pad:end-pad);
D=D(1+pad:end-pad,1+pad:end-pad,1+pad:end-pad);
T=T(1+pad:end-pad,1+pad:end-pad,1+pad:end-pad);
S=S(1+pad:end-pad,1+pad:end-pad,1+pad:end-pad);

try
    D(KT==0)=max(D(KT));
catch
    D(KT==0)=max(D(KT));
end
%D=double((max(D(:))-D)./max(D(:)));

disp(exit_flag)
end