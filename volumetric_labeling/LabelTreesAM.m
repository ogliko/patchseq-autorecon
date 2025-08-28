% This function finds trees in a directed or undirected AM and returns a
% labeled AMlbl.

function AMlbl = LabelTreesAM(AM)

AM = spones(AM+AM');
AMlbl=double(AM);

AV = find(sum(AM));
if ~isempty(AV)
    startV=AV(1);
    TreeLabel=1;
end

while ~isempty(AV)
    startVnew=find(sum(AM(startV,:),1));
    if ~isempty(startVnew)
        AMlbl(startV,startVnew)=AM(startV,startVnew).*TreeLabel;
        AMlbl(startVnew,startV)=AM(startVnew,startV).*TreeLabel;
        AM(startV,startVnew)=0;
        AM(startVnew,startV)=0;
        startV=startVnew;
    else
        AV=find(sum(AM));
        if ~isempty(AV)
            startV=AV(1);
            TreeLabel=TreeLabel+1;
        end
    end
end

