% This function converts AMlbl r R format to swc. Reduction done during image 
% loading is inverted. AMlbl must not contain loops. 

function swc_all = AM2swc(AMlbl,r,R,reduction_x,reduction_y,reduction_z) 

rem_ind=(sum(AMlbl)==0);
AMlbl(rem_ind,:)=[];
AMlbl(:,rem_ind)=[];
r(rem_ind,:)=[];
R(rem_ind)=[];

if isempty(AMlbl)
    swc_all=0;
else
    swc_all=AM2swc_temp(AMlbl,r,R,reduction_x,reduction_y,reduction_z);
end

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
function swc = AM2swc_temp(AMlbl,r,R,reduction_x,reduction_y,reduction_z) 

r=r-0.5; % r ranges from 0.5 to sizeIm+0.5 in Matlab and [0 sizeIm] in Java and SWC

L=unique(AMlbl(AMlbl>0));
swc=zeros(sum(AMlbl(:)>0)/2+length(L),7);
swc(:,1)=(1:length(swc(:,1)));
swc(:,2)=10;

Current_id=1;
for i=1:length(L)
    AMtree=(AMlbl==L(i));
    Current_vertex=find(sum(AMtree,1)==1,1);
    if ~isempty(Current_vertex)
        swc(Current_id,7)=-1;
        swc(Current_id,3:5)=r(Current_vertex,:);
        swc(Current_id,6)=R(Current_vertex);
        Active_Roots=[];
        Active_Roots_pid=[];
        
        while nnz(AMtree)>0 || ~isempty(Active_Roots)
            Current_id=Current_id+1;
            Next_verts=(AMtree(Current_vertex,:));
            Neighb=sum(Next_verts);
            if Neighb==0
                if ~isempty(Active_Roots)
                    Next_vertex=Active_Roots(1);
                    Current_vertex=Next_vertex;
                    swc(Current_id,7)=Active_Roots_pid(1);
                    swc(Current_id,3:5)=r(Current_vertex,:);
                    swc(Current_id,6)=R(Current_vertex);
                    Active_Roots(1)=[];
                    Active_Roots_pid(1)=[];
                else
                    Current_vertex=find(sum(AMtree,1)==1,1);
                    swc(Current_id,7)=-1;
                    swc(Current_id,3:5)=r(Current_vertex,:);
                    swc(Current_id,6)=R(Current_vertex);
                end
            elseif Neighb==1
                Next_vertex=find(Next_verts,1);
                AMtree(Current_vertex,Next_vertex)=0;
                AMtree(Next_vertex,Current_vertex)=0;
                Current_vertex=Next_vertex;
                swc(Current_id,7)=Current_id-1;
                swc(Current_id,3:5)=r(Current_vertex,:);
                swc(Current_id,6)=R(Current_vertex);
            else
                Next_vertex=find(Next_verts,1);
                AMtree(Current_vertex,Next_verts)=0;
                AMtree(Next_verts,Current_vertex)=0;
                Active_Roots=[Active_Roots,find(Next_verts,Neighb-1,'last')];
                Active_Roots_pid=[Active_Roots_pid,(Current_id-1).*ones(1,Neighb-1)];
                Current_vertex=Next_vertex;
                swc(Current_id,7)=Current_id-1;
                swc(Current_id,3:5)=r(Current_vertex,:);
                swc(Current_id,6)=R(Current_vertex);
            end
        end
        Current_id=Current_id+1;
    end
end

swc(:,3:4)=[swc(:,4),swc(:,3)];
swc(:,3)=swc(:,3).*reduction_y;
swc(:,4)=swc(:,4).*reduction_y;
swc(:,5)=swc(:,5).*reduction_z;
swc(:,6)=swc(:,6).*(reduction_x*reduction_y*reduction_z).^(1/3);