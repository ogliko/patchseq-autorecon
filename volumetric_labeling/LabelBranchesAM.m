% This function labels individual branches in AM by using Depth First Search.
% The function works even when there are several disconnected trees in the AM

function AMlbl = LabelBranchesAM(AM) 

AM=spones(AM+AM'); 
AM=AM-diag(diag(AM));
Remaining=find(sum(AM,1)==1 | sum(AM,1)>2);

AMlbl=AM; 
AMlbl(AMlbl==1)=NaN;
CurrentLabel=1;

while sum(isnan(AMlbl(:)))>0
    if isempty(Remaining)
        Remaining=find(isnan(sum(AMlbl,1)),1);
    end
    BegVert=Remaining(1);
    NeighVert=find(isnan(AMlbl(BegVert,:)),1);
    AMlbl(BegVert,NeighVert)=CurrentLabel;
    AMlbl(NeighVert,BegVert)=CurrentLabel;
    if sum(isnan(AMlbl(Remaining(1),:)))==0
        Remaining(1)=[];
    end
    
    while sum(AM(NeighVert,:))==2 && isnan(sum(AMlbl(NeighVert,:)))
        BegVert=NeighVert;
        NeighVert=find(isnan(AMlbl(BegVert,:)),1);
        AMlbl(BegVert,NeighVert)=CurrentLabel;
        AMlbl(NeighVert,BegVert)=CurrentLabel;
    end
    if sum(isnan(AMlbl(NeighVert,:)))==0
        Remaining(Remaining==NeighVert)=[];
    end
    CurrentLabel=CurrentLabel+1;
end

