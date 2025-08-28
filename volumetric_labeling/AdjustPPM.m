% This function adjusts the number of points per micrometer of the trace (ppm).
% Input can be in the form of AM, AMlbl for branches, or AMlbl for trees
% The output is always in the form of AMlbl for trees

function [AMlbl,r,R] = AdjustPPM(AM,r,R,ppm)

AM=spones(AM+AM');
AMlbl = LabelBranchesAM(AM);
leng=size(AMlbl,1);

L=unique(AMlbl(AMlbl>0));
Nvert=zeros(1,length(L));
for i=1:length(L)
    [e1,e2]=find(AMlbl==L(i));
    lll=sum((r(e1,:)-r(e2,:)).^2,2).^0.5;
    Nvert(i)=ceil(sum(lll)/2*ppm)+1;
end
degree=sum(AM,1);
N_new=sum(Nvert)-sum((degree-1).*(degree>2));
N_new_interm=N_new-sum(degree==1)-sum(degree>2);
AMlbl(end+N_new_interm,end+N_new_interm)=0;
r=[r;zeros(N_new_interm,3)];
R=[R;zeros(N_new_interm,1)];

for i=1:length(L)
    [e1, ~]=find(AMlbl==L(i));
    if ~isempty(e1)
        e1=unique(e1);
        endp=e1(sum(AM(:,e1))==1 | sum(AM(:,e1))>=3);
        
        if isempty(endp) % isolated loop
            endp=e1(1);
            r_branch=zeros(length(e1)+1,3);
            r_branch(1,:)=r(endp(1),:);
            r_branch(end,:)=r(endp(1),:);
            R_branch=zeros(length(e1)+1,1);
            R_branch(1)=R(endp(1));
            R_branch(end)=R(endp(1));
        elseif length(endp)==1 % terminal loop
            r_branch=zeros(length(e1)+1,3);
            r_branch(1,:)=r(endp(1),:);
            r_branch(end,:)=r(endp(1),:);
            R_branch=zeros(length(e1)+1,1);
            R_branch(1)=R(endp(1));
            R_branch(end)=R(endp(1));
        else
            r_branch=zeros(length(e1),3);
            r_branch(1,:)=r(endp(1),:);
            r_branch(end,:)=r(endp(2),:);
            R_branch=zeros(length(e1),1);
            R_branch(1)=R(endp(1));
            R_branch(end)=R(endp(2));
        end
        startp=endp(1);
        if length(e1)>2
            for j=2:length(e1)-length(endp)+1
                nextp=find(AMlbl(startp,:)==L(i),1,'first');
                r_branch(j,:)=r(nextp,:);
                R_branch(j)=R(nextp);
                AMlbl(nextp,startp)=0;
                AMlbl(startp,nextp)=0;
                startp=nextp;
            end
            if length(endp)==1 % terminal loop
                AMlbl(endp(1),startp)=0;
                AMlbl(startp,endp(1))=0;
            else
                AMlbl(endp(2),startp)=0;
                AMlbl(startp,endp(2))=0;
            end
        elseif length(e1)==2
            AMlbl(endp(2),endp(1))=0;
            AMlbl(endp(1),endp(2))=0;
        end
        
        lll=sum((r_branch(2:end,:)-r_branch(1:end-1,:)).^2,2).^0.5;
        cumlll=cumsum(lll);
        
        N_interm=ceil(sum(lll)*ppm)-1;
        if N_interm==0 && length(endp)>1 % not a terminal or isolated loop
            AMlbl(endp(2),endp(1))=L(i);
            AMlbl(endp(1),endp(2))=L(i);
        elseif N_interm>=1
            r_interm=zeros(N_interm,3);
            R_interm=zeros(N_interm,1);
            for j=1:N_interm
                temp_ind=find(cumlll>sum(lll)/(N_interm+1)*j,1,'first');
                r_interm(j,:)=r_branch(temp_ind,:)+(r_branch(temp_ind+1,:)-r_branch(temp_ind,:)).*(1-(cumlll(temp_ind)-sum(lll)/(N_interm+1)*j)./lll(temp_ind));
                R_interm(j)=R_branch(temp_ind)+(R_branch(temp_ind+1)-R_branch(temp_ind)).*(1-(cumlll(temp_ind)-sum(lll)/(N_interm+1)*j)./lll(temp_ind));
            end
            
            if leng+N_interm>size(AMlbl,1)
                AMlbl(leng+N_interm,leng+N_interm)=0;
            end
            AMlbl=AMlbl+sparse(leng+[1:N_interm-1,2:N_interm],leng+[2:N_interm,1:N_interm-1],L(i),size(AMlbl,1),size(AMlbl,2));
            AMlbl(leng+1,endp(1))=L(i);
            AMlbl(endp(1),leng+1)=L(i);
            if length(endp)==1 % terminal loop
                AMlbl(leng+N_interm,endp(1))=L(i);
                AMlbl(endp(1),leng+N_interm)=L(i);
            else
                AMlbl(leng+N_interm,endp(2))=L(i);
                AMlbl(endp(2),leng+N_interm)=L(i);
            end
            
            r(leng+1:leng+N_interm,:)=r_interm;
            R(leng+1:leng+N_interm)=R_interm;
            leng=leng+N_interm;
        end
    end
end

rem=(sum(AMlbl,1)==0);
AMlbl(rem,:)=[];
AMlbl(:,rem)=[];
r(rem,:)=[];
R(rem)=[];

AMlbl=LabelTreesAM(AMlbl);
%disp('Point density is adjusted.')

