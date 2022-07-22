%
% Authors: Uygar Sumbul, Olga Gliko
% 
% Allen Institute for Brain Science
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rgcAnalyzer(outdir,arborFileName,swcUnitType, surfaceFilenames,voxelRes,conformalJump, avg_layer_thickness, ...
                Borders, layer_name, units_swc, nHeader, scalefactor, sp, extstr)
% create 2D representations of individual neurons (manage below wm nodes)
% nodes(:,1) = x_swc, nodes(:,2) = z_swc, nodes(:,3) = -y_swc

border_idx = 1:numel(Borders);
for i=1:numel(Borders)
    border_idx(i) = find(strcmp(Borders{i}, layer_name));
end

%read and plot layers
pia_xyz = open_annotationFile(surfaceFilenames{1}, units_swc, scalefactor);
wm_xyz = open_annotationFile(surfaceFilenames{2}, units_swc, scalefactor);

[nodes,edges,radii,nodeTypes,abort] = readArborTrace(arborFileName, swcUnitType, nHeader, units_swc, scalefactor);
disp('arbor trace is read')

warpedArbor = neuronDeformer(nodes, edges, radii, surfaceFilenames, avg_layer_thickness(border_idx), layer_name, voxelRes, conformalJump, units_swc, scalefactor );
assigned_layers = warpedArbor.layers;
warpedArbor.edges=edges;
warpedArbor.radii=radii;
disp('warpedArbor is generated')

% replaced 'arborFilename' with 'outdir + fname +fext'
[~,fname,fext] = fileparts(arborFileName);
csvwrite(strcat(outdir, fname, '_', num2str(length(surfaceFilenames)), '_nodes.csv'), [nodeTypes, nodes]);

somaNodes = find(nodeTypes == 1);
somaX = mean(warpedArbor.nodes(somaNodes,1));
somaY = mean(warpedArbor.nodes(somaNodes,2)); 
somaZ = mean(warpedArbor.nodes(somaNodes,3));
somaRadius = mean(radii(somaNodes));

na_idx = find(isnan(warpedArbor.nodes(:,3))) ;
if (numel(na_idx)>0) disp(strcat('warning : NAs in warpedArbor, outside of bounding mesh :', num2str(numel(na_idx)))); end
Nna=numel(na_idx);
idx_bwm = find(warpedArbor.idx_bwm == 1);
idx_ratio = find(warpedArbor.nodes(:,3)>0.9 & warpedArbor.idx_bwm == 0);
ycoeff = mean(warpedArbor.nodes(idx_ratio,3)./nodes(idx_ratio,3));
warpedArbor.nodes(idx_bwm,3) = nodes(idx_bwm,3)*ycoeff;
csvwrite(strcat(outdir, fname, '_', num2str(length(surfaceFilenames)), '_warpedArbor_nodes.csv'), [double(nodeTypes), warpedArbor.nodes]);

nodes = warpedArbor.nodes;
edges = warpedArbor.edges;

% put the soma at the origin
nodes(:,1) = nodes(:,1) - nodes(1,1); % swc_x
nodes(:,2) = nodes(:,2) - nodes(1,2); % swc_z
        
% scale dimensions
scale_xy = 1000; % swc xz
nodes(:,1) = nodes(:,1)/scale_xy; % swc_x
nodes(:,2) = nodes(:,2)/scale_xy; % swc_z
nodes(:,3) = nodes(:,3) - 0.5; % swc_y range change from [0,1] to [-0.5,0.5]

% save nodes before calculating y vs r
csvwrite(strcat(outdir, fname, '_', num2str(length(surfaceFilenames)), '_transformedArbor_nodes.csv'), [double(nodeTypes) nodes]);

% calculate 1D histogram
% histbins = hist(nodes(:,3),[1/120:1/120:1]-0.5-1/240); % swc_y
% exclude nodes below wm for hist calculation
idx_awm = find(nodes(:,3)<=0.5);
histbins = hist(nodes(idx_awm,3),[1/120:1/120:1]-0.5-1/240);
csvwrite(strcat(outdir, 'hist1d', extstr, '\', 'hist1d_', sp, '.csv'), histbins');

% calculate 2D histogram of y vs r

% convert cartesian coordinates to polar coordinates
[theta,r] = cart2pol(nodes(:,1),nodes(:,2));
disp(["r", min(r), max(r)]);

xe = 4; % # of bins in x
ye = 120; % # of bins in y

Xedges = [0:1/xe:1]*0.5; 
Yedges = [0:1/ye:1]-0.5; 

figure(7);
h = histogram2(r, nodes(:,3), Xedges, Yedges, 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on');

hist2d = h.Values; %dimensions [r y]
hist2d = hist2d.'; %dimensions[y r]
rs = zeros(1,size(hist2d,2));
for i=1:size(hist2d,2)
    rs(i) = pi*(Xedges(i+1)^2-Xedges(i)^2);
end
hist2d = hist2d./rs; %normalize by ring surface
disp(["hist2d", max(max(hist2d))]);

% save hist2d as csv and tif files
filename = strcat('hist2d_', string(ye), 'x', string(xe), '_', sp); 
csvwrite(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '\', filename, '.csv'), hist2d);
imwrite(uint16(hist2d/10),char(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '_tifs\', filename, '.tif')),'tif');

%create 2d hist for separate node types
select_types = [2 3]; % aspiny neurons
% select_types = [3 4]; % spiny neurons

for i=1:numel(select_types)
    disp(select_types(i))
    if select_types(i) == 2
        ext_nodes = '_axon_';
    elseif select_types(i) == 3
        ext_nodes = '_dendrite_'; % aspiny 
%         ext_nodes = '_basal_'; % spiny
    elseif select_types(i) == 4
        ext_nodes = '_apical_';
    end
    disp(ext_nodes)

    % soma node is included to ensure connected nodes
    select = find(nodeTypes==1 | nodeTypes==select_types(i));

    % calculate 1d histogram
    % exclude nodes below wm for hist calculation
    idx_awm = find((nodeTypes==1 | nodeTypes==select_types(i)) & nodes(:,3)<=0.5);
    histbins = hist(nodes(idx_awm,3),[1/120:1/120:1]-0.5-1/240);
    csvwrite(strcat(outdir, 'hist1d', extstr, '\', 'hist1d', ext_nodes, sp, '.csv'), histbins');
    clear histbins
    
    % calculate 2d histogram
    figure(9);
    h = histogram2(r(select), nodes(select,3), Xedges, Yedges, 'DisplayStyle', 'tile', 'ShowEmptyBins', 'on');

    hist2d = h.Values; %dimensions [r y]
    hist2d = hist2d.'; %dimensions[y r]
    hist2d = hist2d./rs; %normalize by ring surface

    % save hist2d as csv and tif files
    filename = strcat('hist2d_', string(ye), 'x', string(xe), ext_nodes, sp); 
    csvwrite(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '\', filename, '.csv'), hist2d);
    imwrite(uint16(hist2d/10),char(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '_tifs\', filename, '.tif')),'tif');
end

%**************************************************************************
% calculate 1d and 2d hist including area below wm
disp('calculate representation including are below wm')

% calculate 1D histogram
histbins = hist(nodes(:,3),[1.2/120:1.2/120:1.2]-0.5-1.2/240); % swc_y replaced 1 to 1.2
csvwrite(strcat(outdir, 'hist1d', extstr, '_bwm\', 'hist1d_', sp, '.csv'), histbins');

% calculate 2D histogram of y vs r
Xedges = [0:1/xe:1]*0.5; 
Yedges = [0:1.2/ye:1.2]-0.5;

hist2d = h.Values; %dimensions [r y]
hist2d = hist2d.'; %dimensions[y r]
rs = zeros(1,size(hist2d,2));
for i=1:size(hist2d,2)
    rs(i) = pi*(Xedges(i+1)^2-Xedges(i)^2);
end
hist2d = hist2d./rs; %normalize by ring surface
disp(["hist2d", max(max(hist2d))]);
% ln_hist2d = log(1 + hist2d); 

% save hist2d as csv and tif files
filename = strcat('hist2d_', string(ye), 'x', string(xe), '_', sp); 
csvwrite(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '_bwm\', filename, '.csv'), hist2d);
imwrite(uint16(hist2d/10),char(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '_bwm_tifs\', filename, '.tif')),'tif');

%create 2d hist for separate node types
select_types = [2 3]; % aspiny neurons
% select_types = [3 4]; % spiny neurons

for i=1:numel(select_types)
    disp(select_types(i))
    if select_types(i) == 2
        ext_nodes = '_axon_';
    elseif select_types(i) == 3
        ext_nodes = '_dendrite_'; % aspiny 
%         ext_nodes = '_basal_'; % spiny
    elseif select_types(i) == 4
        ext_nodes = '_apical_';
    end
    disp(ext_nodes)

    % soma node is included to ensure connected nodes
    select = find(nodeTypes==1 | nodeTypes==select_types(i));

    % calculate 1d histogram
    histbins = hist(nodes(select,3),[1.2/120:1.2/120:1.2]-0.5-1.2/240);
    csvwrite(strcat(outdir, 'hist1d', extstr, '_bwm\', 'hist1d', ext_nodes, sp, '.csv'), histbins');
    
    hist2d = h.Values; %dimensions [r y]
    hist2d = hist2d.'; %dimensions[y r]
    hist2d = hist2d./rs; %normalize by ring surface
  
    % save hist2d as csv and tif files
    filename = strcat('hist2d_', string(ye), 'x', string(xe), ext_nodes, sp); 
    csvwrite(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '_bwm\', filename, '.csv'), hist2d);
    imwrite(uint16(hist2d/10),char(strcat(outdir, 'hist2d_', string(ye), 'x', string(xe), extstr, '_bwm_tifs\', filename, '.tif')),'tif');
end
disp('end')

function layer = open_annotationFile(annotationFilename, units, scalefactor)
fdi = fopen(annotationFilename);
%disp(annotationFilename);
if (fdi < 0)
    disp(annotationFilename)
else
    readin = textscan(fdi,'%f%f%f','delimiter', ',', 'headerLines', 1);
    fclose(fdi);
end
if strcmp(units, "micron")
    x = round(readin{1}/scalefactor) ;
    y = round(readin{3}/scalefactor) ;
    z = -round(readin{2}/scalefactor) ;
end
if strcmp(units, "pixels")
    x = round(readin{1}) ;
    y = round(readin{3}) ;
    z = -round(readin{2}) ;
end
layer = [x y z];

function nonna = remove_na (nodes)
[nrow,ncol] = size(nodes) ;
na_idx = [];
for i=1:ncol
    na_idx_i = find(isnan(nodes(:,i)));
    na_idx = [na_idx na_idx_i];
end
na_idx = unique(na_idx) ;
nonna = nodes;
nonna(na_idx,:) = [];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function warpedArbor = neuronDeformer(nodes,edges,radii,surfaceFilenames,commonDepths,layers_str,voxelRes,conformalJump,layerunits, scalefactor)
% read the arbor trace file - add 1 to node positions because FIJI format for arbor tracing starts from 0
arborBoundaries(1) = min(nodes(:,1)); arborBoundaries(2) = max(nodes(:,1));
arborBoundaries(3) = min(nodes(:,2)); arborBoundaries(4) = max(nodes(:,2));
%generate the SAC surfaces from annotations
for kk = 1:numel(surfaceFilenames)
    %disp(surfaceFilenames{kk});
    surfaceMeshes{kk} = fitSurfaceToLayerAnnotation(surfaceFilenames{kk}, layerunits, scalefactor);
end
% find conformal maps of the ChAT surfaces onto the median plane
surfaceMapping1 = calcWarpedSurfaces(surfaceMeshes,commonDepths,arborBoundaries,conformalJump);
warpedArbor    = calcWarpedArbor(nodes,edges,radii,surfaceMapping1,voxelRes,conformalJump,layers_str);
% warpedArbor = 1;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function vzmesh = fitSurfaceToLayerAnnotation(annotationFilename, units, scalefactor)

fdi = fopen(annotationFilename);
%disp(annotationFilename);
if (fdi < 0)
    disp(annotationFilename)
else
    readin = textscan(fdi,'%f%f%f','delimiter', ',', 'headerLines', 1);
    fclose(fdi);
end
if strcmp(units, "micron")
    x = round(readin{1}/scalefactor) ;
    y = round(readin{3}/scalefactor) ;
    z = -round(readin{2}/scalefactor) ;
end
if strcmp(units, "pixels")
    x = round(readin{1}) ;
    y = round(readin{3}) ;
    z = -round(readin{2}) ;
end

% find the maximum boundaries
xMax = max(x); yMax = max(y);
xMin = min(x); yMin = min(y);
disp(xMin);
disp(xMax);
disp(yMin);
disp(yMax);
% use the correspondence points to calculate local transforms and use those local transforms to map points on the arbor
vzmesh               = griddata(y, x, z, 1:yMax, [1:xMax]', 'cubic');
vzmeshNearest        = griddata(y, x, z, 1:yMax, [1:xMax]', 'nearest');
% vzmesh               = griddata(y, x, z, yMin:yMax, [xMin:xMax]', 'cubic');
% vzmeshNearest        = griddata(y, x, z, yMin:yMax, [xMin:xMax]', 'nearest');
extrapPoints         = isnan(vzmesh);
vzmesh(extrapPoints) = vzmeshNearest(extrapPoints);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function surfaceMapping = calcWarpedSurfaces(surfaceMeshes,commonDepths,arborBoundaries,conformalJump)

% surfaceMeshes : cell variable carrying the mesh for each surface in a separate cell

mainDiagDists   = cell(numel(surfaceMeshes), 1);
skewDiagDists   = cell(numel(surfaceMeshes), 1);
mappedPositions = cell(numel(surfaceMeshes), 1);
[meshSizesX, meshSizesY]=cellfun(@size, surfaceMeshes);
minXpos = arborBoundaries(1); maxXpos = arborBoundaries(2); minYpos = arborBoundaries(3); maxYpos = arborBoundaries(4);
% retain the minimum grid of surface points, where grid resolution is determined by conformalJump
thisx   = [max(minXpos-conformalJump,1):conformalJump:min(maxXpos+conformalJump,min(meshSizesX))];
thisy   = [max(minYpos-conformalJump,1):conformalJump:min(maxYpos+conformalJump,min(meshSizesY))];
for kk = 1:numel(surfaceMeshes)
    % retain the minimum grid
    % calculate the traveling distances on the diagonals of the surfaces - this must be changed to Dijkstra's algorithm for exact results
    [main, skew]      = calculateDiagLength(thisx,thisy,surfaceMeshes{kk}(thisx, thisy));
    mainDiagDists{kk} = main;
    skewDiagDists{kk} = skew;
end
% average the diagonal distances on the surfaces for more stability against band tracing errors - not ideal
mainDiagDist = mean(cell2mat(mainDiagDists)); skewDiagDist = mean(cell2mat(skewDiagDists));
% quasi-conformally map individual surfaces to planes
for kk = 1:numel(surfaceMeshes)
    mappedPositions{kk} = conformalMap_indepFixedDiagonals(mainDiagDist,skewDiagDist,thisx,thisy,surfaceMeshes{kk}(thisx, thisy));
end
% align the two independently mapped surfaces so their flattest regions are registered to each other
xborders = [thisx(1) thisx(end)]; yborders = [thisy(1) thisy(end)];
mappedPositions = alignMappedSurfaces(surfaceMeshes,mappedPositions,xborders,yborders,conformalJump);
% return original and mapped surfaces with grid information
surfaceMapping.mappedPositions = mappedPositions;
surfaceMapping.surfaceMeshes   = surfaceMeshes;
surfaceMapping.thisx           = thisx;
surfaceMapping.thisy           = thisy;
surfaceMapping.commonDepths    = commonDepths;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function warpedArbor = calcWarpedArbor(nodes,edges,radii,surfaceMapping,voxelDim,conformalJump,layers_str)
% voxelDim: physical size of voxels in um, 1x3

% generate correspondence points for the points on the surfaces
thisx                   = surfaceMapping.thisx;
thisy                   = surfaceMapping.thisy;
[tmpymesh,tmpxmesh]     = meshgrid([thisy(1):conformalJump:thisy(end)],[thisx(1):conformalJump:thisx(end)]);
allInputs               = zeros(numel(tmpymesh)*numel(surfaceMapping), 3);
allOutputs              = allInputs;
warpedArbor.meshMedians = zeros(1, numel(surfaceMapping));
idx                     = 1;

% perform the warping (deformation) of the nodse of interest
for kk = 1:numel(surfaceMapping.commonDepths)
    thisMesh                                 = surfaceMapping.surfaceMeshes{kk};
    thisMesh                                 = thisMesh(thisx(1):conformalJump:thisx(end),thisy(1):conformalJump:thisy(end));
    allInputs(idx:idx+numel(tmpymesh)-1, :)  = [tmpxmesh(:) tmpymesh(:) thisMesh(:)];
    thisDepth                                = surfaceMapping.commonDepths(kk);
    allOutputs(idx:idx+numel(tmpymesh)-1, :) = [surfaceMapping.mappedPositions{kk} thisDepth*ones(size(surfaceMapping.mappedPositions{kk},1),1)];
    idx                                      = idx+numel(tmpymesh);
    warpedArbor.meshMedians(kk)              = median(thisMesh(:)); % return median surface positions in cortical depth
end

%for nodes outside bounding mesh, replacing z coordinate with median
%surface position of wm in cortical depth (z) doesn't work well
%instead, for each node replace with approximate surface position of wm
idx_bwm = zeros(size(nodes,1),1); % index for nodes below wm
for i=1:size(nodes,1)
    [~,idx1] = min(abs(thisx - nodes(i,1)));
    if nodes(i,3) < thisMesh(idx1,1)
        idx_bwm(i) = 1;
        if thisMesh(idx1 + 1,1) > thisMesh(idx1 - 1,1)
            nodes(i,3) = thisMesh(idx1 + 1,1);
        else
            nodes(i,3) = thisMesh(idx1 - 1,1);
        end
    end
end
warpedArbor.idx_bwm = idx_bwm;

% use the correspondence points to calculate local transforms and use those local transforms to map points on the arbor
newCoord1 = griddata(allInputs(:,1), allInputs(:,2), allInputs(:,3), allOutputs(:,1), nodes(:,1), nodes(:,2), nodes(:,3), 'linear');
newCoord2 = griddata(allInputs(:,1), allInputs(:,2), allInputs(:,3), allOutputs(:,2), nodes(:,1), nodes(:,2), nodes(:,3), 'linear');
newCoord3 = griddata(allInputs(:,1), allInputs(:,2), allInputs(:,3), allOutputs(:,3), nodes(:,1), nodes(:,2), nodes(:,3), 'linear');
nodes = [newCoord1 newCoord2 newCoord3];

% assign layer based on newCoord3 w.r.t. surfaceMapping.commonDepths
layers=zeros(numel(newCoord3), numel(surfaceMapping.commonDepths));
for kk = 1:(numel(surfaceMapping.commonDepths)-1)
    if kk==1 layers(:,kk) = newCoord3 < surfaceMapping.commonDepths(kk+1) ;
    else layers(:,kk) = newCoord3 > surfaceMapping.commonDepths(kk)  & newCoord3 < surfaceMapping.commonDepths(kk+1) ;
    end
end

layers_assigned = repmat("NA",numel(newCoord3),1);
next_layer = "NA";
for kk=1:numel(newCoord3)
    kk_layer = find(layers(kk,:));
    if (numel(kk_layer)>0) 
        layers_assigned(kk) = layers_str(kk_layer);
    else
        layers_assigned(kk) = next_layer;
    end
    next_layer = layers_assigned(kk);
end
warpedArbor.layers = layers_assigned;

disp('layers are assigned')
%tabulate(layers_assigned) % requires Statistics and Machine Learning Toolbox
% switch to physical dimensions (in um)
nodes(:,1) = nodes(:,1)*voxelDim(1); nodes(:,2) = nodes(:,2)*voxelDim(2); nodes(:,3) = nodes(:,3)*voxelDim(3);
warpedArbor.nodes = nodes;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nodes,edges,radii,nodeTypes,abort] = readArborTrace(fileName, validNodeTypes,nheaderLines, units, scalefactor)
abort = false; nodes = []; edges = []; nodeTypes = [];
validNodeTypes = setdiff(validNodeTypes,1); % 1 is for soma

% read the SWC file
fdi = fopen(fileName);
readin = textscan(fdi, '%u%d%f%f%f%f%d','headerLines', nheaderLines);
fclose(fdi);

nodeID = readin{1} ;
nodeType = readin{2} ;

if strcmp(units, "micron")
    xPos = round(readin{3}/scalefactor) ;
    yPos = round(readin{5}/scalefactor) ;
    zPos = -round(readin{4}/scalefactor) ;
    
end
if strcmp(units, "pixels")
    xPos = round(readin{3}) ;
    yPos = round(readin{5}) ;
    zPos = -round(readin{4}) ;
end

radii= readin{6} ;
parentNodeID= readin{7} ;

% every tree should start from a node of type 1 (soma)
% nodeType(find(parentNodeID==-1))=1; %do not add disconnected segments first nodes

% find the first soma node in the list (more than one node can be labeled as soma)
firstSomaNode = find(nodeType == 1 & parentNodeID == -1, 1);
disp('firstSomaNode:')
disp(firstSomaNode)

% find the average position of all the soma nodes, and assign it as THE soma node position
somaNodes = find(nodeType == 1);
somaX = mean(xPos(somaNodes)); somaY = mean(yPos(somaNodes)); somaZ = mean(zPos(somaNodes));
somaRadius = mean(radii(somaNodes));
xPos(firstSomaNode) = somaX; yPos(firstSomaNode) = somaY; zPos(firstSomaNode) = somaZ;
radii(firstSomaNode) = somaRadius;

% change parenthood so that there is a single soma parent
parentNodeID(ismember(parentNodeID,somaNodes)) = firstSomaNode;

% delete all the soma nodes except for the firstSomaNode
nodesToDelete = setdiff(somaNodes,firstSomaNode);
nodeID(nodesToDelete)=[]; nodeType(nodesToDelete)=[];
xPos(nodesToDelete)=[]; yPos(nodesToDelete)=[]; zPos(nodesToDelete)=[];
radii(nodesToDelete)=[]; parentNodeID(nodesToDelete)=[];

% reassign node IDs due to deletions

for kk = 1:(numel(nodeID)-1)
    while ~any(nodeID==kk)
        kk-1;
        nodeID(nodeID>kk) = nodeID(nodeID>kk)-1;
        parentNodeID(parentNodeID>kk) = parentNodeID(parentNodeID>kk)-1;
    end
end


% of all the nodes, retain the ones indicated in validNodeTypes
% ensure connectedness of the tree if a child is marked as valid but not some of its ancestors

validNodes = nodeID(ismember(nodeType,validNodeTypes));
additionalValidNodes = [];
for kk = 1:numel(validNodes)  
    %kk
    thisParentNodeID = parentNodeID(validNodes(kk)) ;
    thisParentNodeType = nodeType(validNodes(kk));
    %disp([kk, thisParentNodeID, thisParentNodeType])
    while ~ismember(thisParentNodeType,validNodeTypes)
        if thisParentNodeType == 1
            break;
        end
        additionalValidNodes = union(additionalValidNodes, thisParentNodeID); nodeType(thisParentNodeID) = validNodeTypes(1);
        thisParentNodeID = parentNodeID(thisParentNodeID); thisParentNodeType = nodeType(thisParentNodeID);
    end
end

% retain the valid nodes only
% the soma node is always a valid node to ensure connectedness of the tree
validNodes = [firstSomaNode; validNodes; additionalValidNodes']; validNodes = unique(validNodes);
nodeID = nodeID(validNodes); nodeType = nodeType(validNodes); parentNodeID = parentNodeID(validNodes);
xPos = xPos(validNodes); yPos = yPos(validNodes); zPos = zPos(validNodes); radii = radii(validNodes);

% reassign node IDs after deletions

    for kk = 1:(numel(nodeID)-1)
        while ~any(nodeID==kk)
            nodeID(nodeID>kk) = nodeID(nodeID>kk)-1;
            parentNodeID(parentNodeID>kk) = parentNodeID(parentNodeID>kk)-1;
        end
    end


% return the resulting tree data
nodes = [xPos yPos zPos];
edges = [nodeID parentNodeID];
% edges(any(edges<1,2),:) = []; % removes nodes with pid=-1 including soma
nodeTypes = nodeType; %nodeTypes = unique(nodeType)';



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function xy = align2dDataWithMainDiagonal(xyz, weights)
if nargin < 2; weights = ones(size(xyz,1),1); end

% find the inertia tensor
inertiaTensor = zeros(3);
inertiaTensor(1,1) = sum(weights .* (xyz(:,2).^2 + xyz(:,3).^2)); inertiaTensor(2,2) = sum(weights .* (xyz(:,1).^2 + xyz(:,3).^2));
inertiaTensor(3,3) = sum(weights .* (xyz(:,1).^2 + xyz(:,2).^2)); inertiaTensor(1,2) = -sum(weights .* xyz(:,1) .* xyz(:,2));
inertiaTensor(1,3) = -sum(weights .* xyz(:,1) .* xyz(:,3)); inertiaTensor(2,3) = -sum(weights .* xyz(:,2) .* xyz(:,3));
inertiaTensor(2,1) = inertiaTensor(1,2); inertiaTensor(3,1) = inertiaTensor(1,3); inertiaTensor(3,2) = inertiaTensor(2,3);

% find the principal axes of the inertia tensor
[principalAxes, evMatrix] = eig(inertiaTensor);
% take the projection of the 1st principle axis onto the xy plane
pA = principalAxes(1:2,1); pA = pA/norm(pA);
% find the rotation to align pA with the x-axis
pA = pA * sign(xyz(1,1:2)*pA);
%pA = pA * sign(weights'*xyz(:,1:2)*pA);
rotAngle1 = -sign(pA(2))*acos(pA(1)); % (negative of) angle with x-axis
rotMatrix1 = cos(rotAngle1)*eye(2) + sin(rotAngle1)*[0 -1;1 0]; rotMatrix1 = sqrt(1/2)*[1 1;-1 1]*rotMatrix1;
xy = xyz(:,1:2)*rotMatrix1';


function [localMass,newNodes] = segmentLengths(nodes,edges)
% assign new nodes at the center of mass of each edge and calculate the mass (length) of each edge
localMass = zeros(size(nodes,1),1); newNodes = zeros(size(nodes,1),3);
for kk=1:size(nodes,1);
    parent = edges(find(edges(:,1)==kk),2);
    if ~isempty(parent)
        localMass(kk) = norm(nodes(parent,:)-nodes(kk,:)); newNodes(kk,:) = (nodes(parent,:)+nodes(kk,:))/2;
    else
        localMass(kk) = 0; newNodes(kk,:) = nodes(kk,:);
    end
end


function interpolated = gridder_KBinZ_NNinXY(zSamples,xSamples,ySamples,density,n,m,repLenX,lpfilt)

% data must be in [-0.5, 0.5]
alphaZ=2; Wz=3; error=1e-3; Sz=ceil(0.91/error/alphaZ); beta=pi*sqrt((Wz/alphaZ*(alphaZ-1/2))^2-0.8);
Gz=alphaZ*n; F_kbZ=besseli(0,beta*sqrt(1-([-1:2/(Sz*Wz):1]).^2)); z=-(alphaZ*n/2):(alphaZ*n/2)-1; F_kbZ=F_kbZ/max(F_kbZ(:));
kbZ=sqrt(alphaZ*n)*sin(sqrt((pi*Wz*z/Gz).^2-beta^2))./sqrt((pi*Wz*z/Gz).^2-beta^2);  % generate Fourier transform of 1d interpolating kernels

% zero out output array in the alpha grid - use a vector representation to be able to use sparse matrix structure
n = alphaZ*n; interpolated = sparse(n*m*m,1);

% convert samples to matrix indices
nz = (n/2+1) + n*zSamples;

% nearest neighbor interpolation in XY results in some frequency aliasing.
% Low-pass filtering to obtain an arbor density estimate will remove aliased frequencies anyway
nxt = min(m,max(1,round((m/2+1)+m*xSamples))); nyt = min(m,max(1,round((m/2+1)+m*ySamples)));

% loop over samples in kernel
for lz = -(Wz-1)/2:(Wz-1)/2,
    nzt = round(nz+lz); zpos=Sz*((nz-nzt)-(-Wz/2))+1; kwz=F_kbZ(round(zpos))'; nzt = min(max(nzt,1),n);
    linearIndices = sub2ind([n m m],nzt,nxt,nyt); %nzt+(nxt-1)*n+(nyt-1)*n*m;   % compute linear indices
    interpolated = interpolated+sparse(linearIndices,1,density.*kwz,n*m*m,1);   % use sparse matrix to turn k-space trajectory into 2D matrix
end

interpolated=reshape(full(interpolated),n,m,m); interpolated([1 n],:,:)=0;   % edges may be due to samples outside the matrix
n = n/alphaZ; interpolated = myifft3(interpolated,n,repLenX,repLenX);        % pre-filter for decimation, low-pass filtering for arbor density
kbZtmp = kbZ((alphaZ-1)*n/2+1:(alphaZ+1)*n/2); deapod = repmat(kbZtmp',[1 repLenX repLenX]); interpolated = interpolated./deapod; % deapodize

% further low-pass filtering. also removes ringing
interpolated=abs(myfft3(interpolated.*lpfilt));





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mainDiagDist, skewDiagDist] = calculateDiagLength(xpos,ypos,VZmesh)
M = size(VZmesh,1); N = size(VZmesh,2);
[ymesh,xmesh] = meshgrid(ypos,xpos);
mainDiagDist = 0; skewDiagDist = 0;
% travel on the diagonals and not necessarily the grid points) and accumulate the 3d distance traveled
if N >= M
    xKnots = interp2(ymesh,xmesh,xmesh, ypos, [xpos(1):(xpos(end)-xpos(1))/(N-1):xpos(end)]');
    yKnots = interp2(ymesh,xmesh,ymesh, ypos, [xpos(1):(xpos(end)-xpos(1))/(N-1):xpos(end)]');
    zKnotsMainDiag = griddata(xmesh(:),ymesh(:),VZmesh(:), [xpos(1):(xpos(end)-xpos(1))/(N-1):xpos(end)]', ypos');
    zKnotsSkewDiag = griddata(xmesh(:),ymesh(:),VZmesh(:), [xpos(1):(xpos(end)-xpos(1))/(N-1):xpos(end)]', ypos(end:-1:1)');
    dxKnots = diag(xKnots); dyKnots = diag(yKnots); mainDiagDist = sum(sqrt(diff(dxKnots).^2 + diff(dyKnots).^2 + diff(zKnotsMainDiag).^2));
    sdxKnots = xKnots(N:N-1:end-1)'; sdyKnots = yKnots(N:N-1:end-1)'; skewDiagDist = sum(sqrt(diff(sdxKnots).^2 + diff(sdyKnots).^2 + diff(zKnotsSkewDiag).^2));
    
else
    xKnots = interp2(ymesh,xmesh,xmesh, [ypos(1):(ypos(end)-ypos(1))/(M-1):ypos(end)], xpos');
    yKnots = interp2(ymesh,xmesh,ymesh, [ypos(1):(ypos(end)-ypos(1))/(M-1):ypos(end)], xpos');
    zKnotsMainDiag = griddata(xmesh(:),ymesh(:),VZmesh(:), xpos', [ypos(1):(ypos(end)-ypos(1))/(M-1):ypos(end)]');
    zKnotsSkewDiag = griddata(xmesh(:),ymesh(:),VZmesh(:), xpos', [ypos(end):-(ypos(end)-ypos(1))/(M-1):ypos(1)]');
    dxKnots = diag(xKnots); dyKnots = diag(yKnots); mainDiagDist = sum(sqrt(diff(dxKnots).^2 + diff(dyKnots).^2 + diff(zKnotsMainDiag).^2));
    sdxKnots = xKnots(M:M-1:end-1)'; sdyKnots = yKnots(M:M-1:end-1)'; skewDiagDist = sum(sqrt(diff(sdxKnots).^2 + diff(sdyKnots).^2 + diff(zKnotsSkewDiag).^2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mappedPositions = alignMappedSurfaces(surfaceMeshes,mappedPositions,validXborders,validYborders,conformalJump,patchSize)
if nargin < 6
    patchSize = 21;
    if nargin < 5
        conformalJump = 1;
    end
end
patchSize = ceil(patchSize/conformalJump);
% calculate the total gradient surfaces
dSurfaces = cell(1, numel(surfaceMeshes));
for kk = 1:numel(surfaceMeshes)
    % pad the surface by one pixel so that the size remains the same after the difference operation
    thisSurface = surfaceMeshes{kk};
    surfaceMeshes{kk} = [[thisSurface 10*max(thisSurface(:))*ones(size(thisSurface,1),1)]; 10*max(thisSurface(:))*ones(1,size(thisSurface,2)+1)];
    % calculate the absolute differences in xy between neighboring pixels
    tmp1 = diff(surfaceMeshes{kk},1,1); tmp1 = tmp1(:,1:end-1); tmp2 = diff(surfaceMeshes{kk},1,2); tmp2 = tmp2(1:end-1,:); thisDsurface = abs(tmp1+i*tmp2);
    % retain the region of interest, with a resolution specified by conformalJump
    dSurfaces{kk} = thisDsurface(validXborders(1):conformalJump:validXborders(2), validYborders(1):conformalJump:validYborders(2));
end
% calculate the cost as the sum of absolute slopes on all surfaces
patchCosts = conv2(sum(cat(3,dSurfaces{:}),3), ones(patchSize),'valid');
% find the minimum cost
[row,col] = find(patchCosts == min(min(patchCosts)),1);
row = row+(patchSize-1)/2; col = col+(patchSize-1)/2;
% calculate and apply the corresponding shift
linearInd = sub2ind(size(dSurfaces{end}),row,col);
reference = mappedPositions{end};

for kk = 1:numel(dSurfaces)-1
    thisPositions       = mappedPositions{kk};
    shift1              = thisPositions(linearInd,1)-reference(linearInd,1);
    thisPositions(:,1)  = thisPositions(:,1) - shift1;
    shift2              = thisPositions(linearInd,2)-reference(linearInd,2);
    thisPositions(:,2)  = thisPositions(:,2) - shift2;
    mappedPositions{kk} = thisPositions;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function mappedPositions = conformalMap_indepFixedDiagonals(mainDiagDist,skewDiagDist,xpos,ypos,VZmesh)
% implements quasi-conformal mapping suggested in
% Levy et al., 'Least squares conformal maps for automatic texture atlas generation', 2002, ACM Transactions on Graphics
M = size(VZmesh,1); N = size(VZmesh,2);
col1 = kron([1;1],[1:M-1]');
temp1 = kron([1;M+1],ones(M-1,1));
temp2 = kron([M+1;M],ones(M-1,1));
oneColumn = [col1 col1+temp1 col1+temp2];
% every pixel is divided into 2 triangles
triangleCount = (2*M-2)*(N-1);
vertexCount = M*N;
triangulation = zeros(triangleCount, 3);
% store the positions of the vertices for each triangle - triangles are oriented consistently
for kk = 1:N-1
    triangulation((kk-1)*(2*M-2)+1:kk*(2*M-2),:) = oneColumn + (kk-1)*M;
end
Mreal = sparse([],[],[],triangleCount,vertexCount,triangleCount*3);
Mimag = sparse([],[],[],triangleCount,vertexCount,triangleCount*3);
% calculate the conformality condition (Riemann's theorem)
for triangle = 1:triangleCount
    for vertex = 1:3
        nodeNumber = triangulation(triangle,vertex);
        xind = rem(nodeNumber-1,M)+1;
        yind = floor((nodeNumber-1)/M)+1;
        trianglePos(vertex,:) = [xpos(xind) ypos(yind) VZmesh(xind,yind)];
    end
    [w1,w2,w3,zeta] = assignLocalCoordinates(trianglePos);
    denominator = sqrt(zeta/2);
    Mreal(triangle,triangulation(triangle,1)) = real(w1)/denominator;
    Mreal(triangle,triangulation(triangle,2)) = real(w2)/denominator;
    Mreal(triangle,triangulation(triangle,3)) = real(w3)/denominator;
    Mimag(triangle,triangulation(triangle,1)) = imag(w1)/denominator;
    Mimag(triangle,triangulation(triangle,2)) = imag(w2)/denominator;
    Mimag(triangle,triangulation(triangle,3)) = imag(w3)/denominator;
end
% minimize the LS error due to conformality condition of mapping triangles into triangles
% take the two fixed points required to solve the system as the corners of the main diagonal
mainDiagXdist = mainDiagDist*M/sqrt(M^2+N^2); mainDiagYdist = mainDiagXdist*N/M;
A = [Mreal(:,2:end-1) -Mimag(:,2:end-1); Mimag(:,2:end-1) Mreal(:,2:end-1)];
b = -[Mreal(:,[1 end]) -Mimag(:,[1 end]); Mimag(:,[1 end]) Mreal(:,[1 end])]*[[xpos(1);xpos(1)+mainDiagXdist];[ypos(1);ypos(1)+mainDiagYdist]];
mappedPositions1 = A\b;
mappedPositions1 = [[xpos(1);mappedPositions1(1:end/2);xpos(1)+mainDiagXdist] [ypos(1);mappedPositions1(1+end/2:end);ypos(1)+mainDiagYdist]];
% take the two fixed points required to solve the system as the corners of the skew diagonal
skewDiagXdist = skewDiagDist*M/sqrt(M^2+N^2); skewDiagYdist = skewDiagXdist*N/M; freeVar = [[1:M-1] [M+1:M*N-M] [M*N-M+2:M*N]];
A = [Mreal(:,freeVar) -Mimag(:,freeVar); Mimag(:,freeVar) Mreal(:,freeVar)];
b = -[Mreal(:,[M M*N-M+1]) -Mimag(:,[M M*N-M+1]); Mimag(:,[M M*N-M+1]) Mreal(:,[M M*N-M+1])]*[[xpos(1)+skewDiagXdist;xpos(1)];[ypos(1);ypos(1)+skewDiagYdist]];
mappedPositions2 = A\b;
mappedPositions2 = [[mappedPositions2(1:M-1);xpos(1)+skewDiagXdist;mappedPositions2(M:M*N-M-1);xpos(1);mappedPositions2(M*N-M:end/2)] ...
    [mappedPositions2(1+end/2:M-1+end/2);ypos(1);mappedPositions2(end/2+M:end/2+M*N-M-1);ypos(1)+skewDiagYdist;mappedPositions2(end/2+M*N-M:end)]];
% take the mean of the two independent solutions and return it as the solution
mappedPositions = (mappedPositions1+mappedPositions2)/2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w1,w2,w3,zeta] = assignLocalCoordinates(triangle)
% triangle is a 3x3 matrix where the rows represent the x,y,z coordinates of the vertices.
% The local triangle is defined on a plane, where the first vertex is at the origin(0,0) and the second vertex is at (0,-d12).
d12 = norm(triangle(1,:)-triangle(2,:));
d13 = norm(triangle(1,:)-triangle(3,:));
d23 = norm(triangle(2,:)-triangle(3,:));
y3 = ((-d12)^2+d13^2-d23^2)/(2*(-d12));
x3 = sqrt(d13^2-y3^2); % provisional

w2 = -x3-i*y3; w1 = x3+i*(y3-(-d12));
zeta = i*(conj(w2)*w1-conj(w1)*w2); % orientation indicator
w3 = i*(-d12);
zeta = abs(real(zeta));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function f=myifft3(F,u,v,w)
% does the centering(shifts) and normalization
% needs a slight modification to actually support rectangular images
if nargin<4
    f=ifftshift(ifftn(ifftshift(F)))*sqrt(prod(size(F)));
else
    f=ifftshift(ifftn(ifftshift(F)))*sqrt(u*v*w);
    zoffset=ceil((size(f,1)-u)/2); xoffset=ceil((size(f,2)-v)/2); yoffset=ceil((size(f,3)-w)/2);
    f=f(zoffset+1:zoffset+u,xoffset+1:xoffset+v,yoffset+1:yoffset+w);
end

function F=myfft3(f,u,v,w)
% does the centering(shifts) and normalization
% needs a slight modification to actually support rectangular images
if nargin<4
    F=fftshift(fftn(fftshift(f)))/sqrt(prod(size(f)));
else
    F=zeros(u,v,w);
    zoffset=(u-size(f,1))/2; xoffset=(v-size(f,2))/2; yoffset=(w-size(f,3))/2;
    F(zoffset+1:zoffset+size(f,1),xoffset+1:xoffset+size(f,2),yoffset+1:yoffset+size(f,3))=f;
    F=fftshift(fftn(fftshift(F)))/sqrt(prod(size(f)));
end


function [zgrid,xgrid,ygrid] = gridfit(x,y,z,xnodes,ynodes,varargin)

%--------------------------%

%Copyright (c) 2006, John D'Errico

%All rights reserved.



%Redistribution and use in source and binary forms, with or without

%modification, are permitted provided that the following conditions are

%met:



%    * Redistributions of source code must retain the above copyright

%      notice, this list of conditions and the following disclaimer.

%    * Redistributions in binary form must reproduce the above copyright

%      notice, this list of conditions and the following disclaimer in

%      the documentation and/or other materials provided with the distribution



%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"

%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE

%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE

%ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE

%LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR

%CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF

%SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS

%INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN

%CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)

%ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE

%POSSIBILITY OF SUCH DAMAGE.

%--------------------------%



% gridfit: estimates a surface on a 2d grid, based on scattered data

%          Replicates are allowed. All methods extrapolate to the grid

%          boundaries. Gridfit uses a modified ridge estimator to

%          generate the surface, where the bias is toward smoothness.

%

%          Gridfit is not an interpolant. Its goal is a smooth surface

%          that approximates your data, but allows you to control the

%          amount of smoothing.

%

% usage #1: zgrid = gridfit(x,y,z,xnodes,ynodes);

% usage #2: [zgrid,xgrid,ygrid] = gridfit(x,y,z,xnodes,ynodes);

% usage #3: zgrid = gridfit(x,y,z,xnodes,ynodes,prop,val,prop,val,...);

%

% Arguments: (input)

%  x,y,z - vectors of equal lengths, containing arbitrary scattered data

%          The only constraint on x and y is they cannot ALL fall on a

%          single line in the x-y plane. Replicate points will be treated

%          in a least squares sense.

%

%          ANY points containing a NaN are ignored in the estimation

%

%  xnodes - vector defining the nodes in the grid in the independent

%          variable (x). xnodes need not be equally spaced. xnodes

%          must completely span the data. If they do not, then the

%          'extend' property is applied, adjusting the first and last

%          nodes to be extended as necessary. See below for a complete

%          description of the 'extend' property.

%

%          If xnodes is a scalar integer, then it specifies the number

%          of equally spaced nodes between the min and max of the data.

%

%  ynodes - vector defining the nodes in the grid in the independent

%          variable (y). ynodes need not be equally spaced.

%

%          If ynodes is a scalar integer, then it specifies the number

%          of equally spaced nodes between the min and max of the data.

%

%          Also see the extend property.

%

%  Additional arguments follow in the form of property/value pairs.

%  Valid properties are:

%    'smoothness', 'interp', 'regularizer', 'solver', 'maxiter'

%    'extend', 'tilesize', 'overlap'

%

%  Any UNAMBIGUOUS shortening (even down to a single letter) is

%  valid for property names. All properties have default values,

%  chosen (I hope) to give a reasonable result out of the box.

%

%   'smoothness' - scalar or vector of length 2 - determines the

%          eventual smoothness of the estimated surface. A larger

%          value here means the surface will be smoother. Smoothness

%          must be a non-negative real number.

%

%          If this parameter is a vector of length 2, then it defines

%          the relative smoothing to be associated with the x and y

%          variables. This allows the user to apply a different amount

%          of smoothing in the x dimension compared to the y dimension.

%

%          Note: the problem is normalized in advance so that a

%          smoothness of 1 MAY generate reasonable results. If you

%          find the result is too smooth, then use a smaller value

%          for this parameter. Likewise, bumpy surfaces suggest use

%          of a larger value. (Sometimes, use of an iterative solver

%          with too small a limit on the maximum number of iterations

%          will result in non-convergence.)

%

%          DEFAULT: 1

%

%

%   'interp' - character, denotes the interpolation scheme used

%          to interpolate the data.

%

%          DEFAULT: 'triangle'

%

%          'bilinear' - use bilinear interpolation within the grid

%                     (also known as tensor product linear interpolation)

%

%          'triangle' - split each cell in the grid into a triangle,

%                     then linear interpolation inside each triangle

%

%          'nearest' - nearest neighbor interpolation. This will

%                     rarely be a good choice, but I included it

%                     as an option for completeness.

%

%

%   'regularizer' - character flag, denotes the regularization

%          paradignm to be used. There are currently three options.

%

%          DEFAULT: 'gradient'

%

%          'diffusion' or 'laplacian' - uses a finite difference

%              approximation to the Laplacian operator (i.e, del^2).

%

%              We can think of the surface as a plate, wherein the

%              bending rigidity of the plate is specified by the user

%              as a number relative to the importance of fidelity to

%              the data. A stiffer plate will result in a smoother

%              surface overall, but fit the data less well. I've

%              modeled a simple plate using the Laplacian, del^2. (A

%              projected enhancement is to do a better job with the

%              plate equations.)

%

%              We can also view the regularizer as a diffusion problem,

%              where the relative thermal conductivity is supplied.

%              Here interpolation is seen as a problem of finding the

%              steady temperature profile in an object, given a set of

%              points held at a fixed temperature. Extrapolation will

%              be linear. Both paradigms are appropriate for a Laplacian

%              regularizer.

%

%          'gradient' - attempts to ensure the gradient is as smooth

%              as possible everywhere. Its subtly different from the

%              'diffusion' option, in that here the directional

%              derivatives are biased to be smooth across cell

%              boundaries in the grid.

%

%              The gradient option uncouples the terms in the Laplacian.

%              Think of it as two coupled PDEs instead of one PDE. Why

%              are they different at all? The terms in the Laplacian

%              can balance each other.

%

%          'springs' - uses a spring model connecting nodes to each

%              other, as well as connecting data points to the nodes

%              in the grid. This choice will cause any extrapolation

%              to be as constant as possible.

%

%              Here the smoothing parameter is the relative stiffness

%              of the springs connecting the nodes to each other compared

%              to the stiffness of a spting connecting the lattice to

%              each data point. Since all springs have a rest length

%              (length at which the spring has zero potential energy)

%              of zero, any extrapolation will be minimized.

%

%          Note: The 'springs' regularizer tends to drag the surface

%          towards the mean of all the data, so too large a smoothing

%          parameter may be a problem.

%

%

%   'solver' - character flag - denotes the solver used for the

%          resulting linear system. Different solvers will have

%          different solution times depending upon the specific

%          problem to be solved. Up to a certain size grid, the

%          direct \ solver will often be speedy, until memory

%          swaps causes problems.

%

%          What solver should you use? Problems with a significant

%          amount of extrapolation should avoid lsqr. \ may be

%          best numerically for small smoothnesss parameters and

%          high extents of extrapolation.

%

%          Large numbers of points will slow down the direct

%          \, but when applied to the normal equations, \ can be

%          quite fast. Since the equations generated by these

%          methods will tend to be well conditioned, the normal

%          equations are not a bad choice of method to use. Beware

%          when a small smoothing parameter is used, since this will

%          make the equations less well conditioned.

%

%          DEFAULT: 'normal'

%

%          '\' - uses matlab's backslash operator to solve the sparse

%                     system. 'backslash' is an alternate name.

%

%          'symmlq' - uses matlab's iterative symmlq solver

%

%          'lsqr' - uses matlab's iterative lsqr solver

%

%          'normal' - uses \ to solve the normal equations.

%

%

%   'maxiter' - only applies to iterative solvers - defines the

%          maximum number of iterations for an iterative solver

%

%          DEFAULT: min(10000,length(xnodes)*length(ynodes))

%

%

%   'extend' - character flag - controls whether the first and last

%          nodes in each dimension are allowed to be adjusted to

%          bound the data, and whether the user will be warned if

%          this was deemed necessary to happen.

%

%          DEFAULT: 'warning'

%

%          'warning' - Adjust the first and/or last node in

%                     x or y if the nodes do not FULLY contain

%                     the data. Issue a warning message to this

%                     effect, telling the amount of adjustment

%                     applied.

%

%          'never'  - Issue an error message when the nodes do

%                     not absolutely contain the data.

%

%          'always' - automatically adjust the first and last

%                     nodes in each dimension if necessary.

%                     No warning is given when this option is set.

%

%

%   'tilesize' - grids which are simply too large to solve for

%          in one single estimation step can be built as a set

%          of tiles. For example, a 1000x1000 grid will require

%          the estimation of 1e6 unknowns. This is likely to

%          require more memory (and time) than you have available.

%          But if your data is dense enough, then you can model

%          it locally using smaller tiles of the grid.

%

%          My recommendation for a reasonable tilesize is

%          roughly 100 to 200. Tiles of this size take only

%          a few seconds to solve normally, so the entire grid

%          can be modeled in a finite amount of time. The minimum

%          tilesize can never be less than 3, although even this

%          size tile is so small as to be ridiculous.

%

%          If your data is so sparse than some tiles contain

%          insufficient data to model, then those tiles will

%          be left as NaNs.

%

%          DEFAULT: inf

%

%

%   'overlap' - Tiles in a grid have some overlap, so they

%          can minimize any problems along the edge of a tile.

%          In this overlapped region, the grid is built using a

%          bi-linear combination of the overlapping tiles.

%

%          The overlap is specified as a fraction of the tile

%          size, so an overlap of 0.20 means there will be a 20%

%          overlap of successive tiles. I do allow a zero overlap,

%          but it must be no more than 1/2.

%

%          0 <= overlap <= 0.5

%

%          Overlap is ignored if the tilesize is greater than the

%          number of nodes in both directions.

%

%          DEFAULT: 0.20

%

%

%   'autoscale' - Some data may have widely different scales on

%          the respective x and y axes. If this happens, then

%          the regularization may experience difficulties.

%

%          autoscale = 'on' will cause gridfit to scale the x

%          and y node intervals to a unit length. This should

%          improve the regularization procedure. The scaling is

%          purely internal.

%

%          autoscale = 'off' will disable automatic scaling

%

%          DEFAULT: 'on'

%

%

% Arguments: (output)

%  zgrid   - (nx,ny) array containing the fitted surface

%

%  xgrid, ygrid - as returned by meshgrid(xnodes,ynodes)

%

%

% Speed considerations:

%  Remember that gridfit must solve a LARGE system of linear

%  equations. There will be as many unknowns as the total

%  number of nodes in the final lattice. While these equations

%  may be sparse, solving a system of 10000 equations may take

%  a second or so. Very large problems may benefit from the

%  iterative solvers or from tiling.

%

%

% Example usage:

%

%  x = rand(100,1);

%  y = rand(100,1);

%  z = exp(x+2*y);

%  xnodes = 0:.1:1;

%  ynodes = 0:.1:1;

%

%  g = gridfit(x,y,z,xnodes,ynodes);

%

% Note: this is equivalent to the following call:

%

%  g = gridfit(x,y,z,xnodes,ynodes, ...
%              'smooth',1, ...
%              'interp','triangle', ...
%              'solver','normal', ...
%              'regularizer','gradient', ...
%              'extend','warning', ...
%              'tilesize',inf);

%

%

% Author: John D'Errico

% e-mail address: woodchips@rochester.rr.com

% Release: 2.0

% Release date: 5/23/06



% set defaults

params.smoothness = 1;

params.interp = 'triangle';

params.regularizer = 'gradient';

params.solver = 'backslash';

params.maxiter = [];

params.extend = 'warning';

params.tilesize = inf;

params.overlap = 0.20;

params.mask = [];

params.autoscale = 'on';

params.xscale = 1;

params.yscale = 1;



% was the params struct supplied?

if ~isempty(varargin)
    
    if isstruct(varargin{1})
        
        % params is only supplied if its a call from tiled_gridfit
        
        params = varargin{1};
        
        if length(varargin)>1
            
            % check for any overrides
            
            params = parse_pv_pairs(params,varargin{2:end});
            
        end
        
    else
        
        % check for any overrides of the defaults
        
        params = parse_pv_pairs(params,varargin);
        
        
        
    end
    
end



% check the parameters for acceptability

params = check_params(params);



% ensure all of x,y,z,xnodes,ynodes are column vectors,

% also drop any NaN data

x=x(:);

y=y(:);

z=z(:);

k = isnan(x) | isnan(y) | isnan(z);

if any(k)
    
    x(k)=[];
    
    y(k)=[];
    
    z(k)=[];
    
end

xmin = min(x);

xmax = max(x);

ymin = min(y);

ymax = max(y);



% did they supply a scalar for the nodes?

if length(xnodes)==1
    
    xnodes = linspace(xmin,xmax,xnodes)';
    
    xnodes(end) = xmax; % make sure it hits the max
    
end

if length(ynodes)==1
    
    ynodes = linspace(ymin,ymax,ynodes)';
    
    ynodes(end) = ymax; % make sure it hits the max
    
end



xnodes=xnodes(:);

ynodes=ynodes(:);

dx = diff(xnodes);

dy = diff(ynodes);

nx = length(xnodes);

ny = length(ynodes);

ngrid = nx*ny;



% set the scaling if autoscale was on

if strcmpi(params.autoscale,'on')
    
    params.xscale = mean(dx);
    
    params.yscale = mean(dy);
    
    params.autoscale = 'off';
    
end



% check to see if any tiling is necessary

if (params.tilesize < max(nx,ny))
    
    % split it into smaller tiles. compute zgrid and ygrid
    
    % at the very end if requested
    
    zgrid = tiled_gridfit(x,y,z,xnodes,ynodes,params);
    
else
    
    % its a single tile.
    
    
    
    % mask must be either an empty array, or a boolean
    
    % aray of the same size as the final grid.
    
    nmask = size(params.mask);
    
    if ~isempty(params.mask) && ((nmask(2)~=nx) || (nmask(1)~=ny))
        
        if ((nmask(2)==ny) || (nmask(1)==nx))
            
            error 'Mask array is probably transposed from proper orientation.'
            
        else
            
            error 'Mask array must be the same size as the final grid.'
            
        end
        
    end
    
    if ~isempty(params.mask)
        
        params.maskflag = 1;
        
    else
        
        params.maskflag = 0;
        
    end
    
    
    
    % default for maxiter?
    
    if isempty(params.maxiter)
        
        params.maxiter = min(10000,nx*ny);
        
    end
    
    
    
    % check lengths of the data
    
    n = length(x);
    
    if (length(y)~=n) || (length(z)~=n)
        
        error 'Data vectors are incompatible in size.'
        
    end
    
    if n<3
        
        error 'Insufficient data for surface estimation.'
        
    end
    
    
    
    % verify the nodes are distinct
    
    if any(diff(xnodes)<=0) || any(diff(ynodes)<=0)
        
        error 'xnodes and ynodes must be monotone increasing'
        
    end
    
    
    
    % do we need to tweak the first or last node in x or y?
    
    if xmin<xnodes(1)
        
        switch params.extend
            
            case 'always'
                
                xnodes(1) = xmin;
                
            case 'warning'
                
                warning('GRIDFIT:extend',['xnodes(1) was decreased by: ',num2str(xnodes(1)-xmin),', new node = ',num2str(xmin)])
                
                xnodes(1) = xmin;
                
            case 'never'
                
                error(['Some x (',num2str(xmin),') falls below xnodes(1) by: ',num2str(xnodes(1)-xmin)])
                
        end
        
    end
    
    if xmax>xnodes(end)
        
        switch params.extend
            
            case 'always'
                
                xnodes(end) = xmax;
                
            case 'warning'
                
                warning('GRIDFIT:extend',['xnodes(end) was increased by: ',num2str(xmax-xnodes(end)),', new node = ',num2str(xmax)])
                
                xnodes(end) = xmax;
                
            case 'never'
                
                error(['Some x (',num2str(xmax),') falls above xnodes(end) by: ',num2str(xmax-xnodes(end))])
                
        end
        
    end
    
    if ymin<ynodes(1)
        
        switch params.extend
            
            case 'always'
                
                ynodes(1) = ymin;
                
            case 'warning'
                
                warning('GRIDFIT:extend',['ynodes(1) was decreased by: ',num2str(ynodes(1)-ymin),', new node = ',num2str(ymin)])
                
                ynodes(1) = ymin;
                
            case 'never'
                
                error(['Some y (',num2str(ymin),') falls below ynodes(1) by: ',num2str(ynodes(1)-ymin)])
                
        end
        
    end
    
    if ymax>ynodes(end)
        
        switch params.extend
            
            case 'always'
                
                ynodes(end) = ymax;
                
            case 'warning'
                
                warning('GRIDFIT:extend',['ynodes(end) was increased by: ',num2str(ymax-ynodes(end)),', new node = ',num2str(ymax)])
                
                ynodes(end) = ymax;
                
            case 'never'
                
                error(['Some y (',num2str(ymax),') falls above ynodes(end) by: ',num2str(ymax-ynodes(end))])
                
        end
        
    end
    
    
    
    % determine which cell in the array each point lies in
    
    [junk,indx] = histc(x,xnodes); %#ok
    
    [junk,indy] = histc(y,ynodes); %#ok
    
    % any point falling at the last node is taken to be
    
    % inside the last cell in x or y.
    
    k=(indx==nx);
    
    indx(k)=indx(k)-1;
    
    k=(indy==ny);
    
    indy(k)=indy(k)-1;
    
    ind = indy + ny*(indx-1);
    
    
    
    % Do we have a mask to apply?
    
    if params.maskflag
        
        % if we do, then we need to ensure that every
        
        % cell with at least one data point also has at
        
        % least all of its corners unmasked.
        
        params.mask(ind) = 1;
        
        params.mask(ind+1) = 1;
        
        params.mask(ind+ny) = 1;
        
        params.mask(ind+ny+1) = 1;
        
    end
    
    
    
    % interpolation equations for each point
    
    tx = min(1,max(0,(x - xnodes(indx))./dx(indx)));
    ty = min(1,max(0,(y - ynodes(indy))./dy(indy)));
    
    % Future enhancement: add cubic interpolant
    
    switch params.interp
        
        case 'triangle'
            
            % linear interpolation inside each triangle
            
            k = (tx > ty);
            
            L = ones(n,1);
            
            L(k) = ny;
            
            
            
            t1 = min(tx,ty);
            
            t2 = max(tx,ty);
            
            A = sparse(repmat((1:n)',1,3),[ind,ind+ny+1,ind+L], [1-t2,t1,t2-t1],n,ngrid);
            
            
            
        case 'nearest'
            
            % nearest neighbor interpolation in a cell
            
            k = round(1-ty) + round(1-tx)*ny;
            
            A = sparse((1:n)',ind+k,ones(n,1),n,ngrid);
            
            
            
        case 'bilinear'
            
            % bilinear interpolation in a cell
            
            A = sparse(repmat((1:n)',1,4),[ind,ind+1,ind+ny,ind+ny+1], [(1-tx).*(1-ty), (1-tx).*ty, tx.*(1-ty), tx.*ty], n,ngrid);
            
            
            
    end
    
    rhs = z;
    
    
    
    % do we have relative smoothing parameters?
    
    if numel(params.smoothness) == 1
        
        % it was scalar, so treat both dimensions equally
        
        smoothparam = params.smoothness;
        
        xyRelativeStiffness = [1;1];
        
    else
        
        % It was a vector, so anisotropy reigns.
        
        % I've already checked that the vector was of length 2
        
        smoothparam = sqrt(prod(params.smoothness));
        
        xyRelativeStiffness = params.smoothness(:)./smoothparam;
        
    end
    
    
    
    % Build regularizer. Add del^4 regularizer one day.
    
    switch params.regularizer
        
        case 'springs'
            
            % zero "rest length" springs
            
            [i,j] = meshgrid(1:nx,1:(ny-1));
            
            ind = j(:) + ny*(i(:)-1);
            
            m = nx*(ny-1);
            
            stiffness = 1./(dy/params.yscale);
            
            Areg = sparse(repmat((1:m)',1,2),[ind,ind+1], ...
                xyRelativeStiffness(2)*stiffness(j(:))*[-1 1], ...
                m,ngrid);
            
            
            
            [i,j] = meshgrid(1:(nx-1),1:ny);
            
            ind = j(:) + ny*(i(:)-1);
            
            m = (nx-1)*ny;
            
            stiffness = 1./(dx/params.xscale);
            
            Areg = [Areg;sparse(repmat((1:m)',1,2),[ind,ind+ny], ...
                xyRelativeStiffness(1)*stiffness(i(:))*[-1 1],m,ngrid)];
            
            
            
            [i,j] = meshgrid(1:(nx-1),1:(ny-1));
            
            ind = j(:) + ny*(i(:)-1);
            
            m = (nx-1)*(ny-1);
            
            stiffness = 1./sqrt((dx(i(:))/params.xscale/xyRelativeStiffness(1)).^2 + ...
                (dy(j(:))/params.yscale/xyRelativeStiffness(2)).^2);
            
            
            
            Areg = [Areg;sparse(repmat((1:m)',1,2),[ind,ind+ny+1], ...
                stiffness*[-1 1],m,ngrid)];
            
            
            
            Areg = [Areg;sparse(repmat((1:m)',1,2),[ind+1,ind+ny], ...
                stiffness*[-1 1],m,ngrid)];
            
            
            
        case {'diffusion' 'laplacian'}
            
            % thermal diffusion using Laplacian (del^2)
            
            [i,j] = meshgrid(1:nx,2:(ny-1));
            
            ind = j(:) + ny*(i(:)-1);
            
            dy1 = dy(j(:)-1)/params.yscale;
            
            dy2 = dy(j(:))/params.yscale;
            
            
            
            Areg = sparse(repmat(ind,1,3),[ind-1,ind,ind+1], ...
                xyRelativeStiffness(2)*[-2./(dy1.*(dy1+dy2)), ...
                2./(dy1.*dy2), -2./(dy2.*(dy1+dy2))],ngrid,ngrid);
            
            
            
            [i,j] = meshgrid(2:(nx-1),1:ny);
            
            ind = j(:) + ny*(i(:)-1);
            
            dx1 = dx(i(:)-1)/params.xscale;
            
            dx2 = dx(i(:))/params.xscale;
            
            
            
            Areg = Areg + sparse(repmat(ind,1,3),[ind-ny,ind,ind+ny], ...
                xyRelativeStiffness(1)*[-2./(dx1.*(dx1+dx2)), ...
                2./(dx1.*dx2), -2./(dx2.*(dx1+dx2))],ngrid,ngrid);
            
            
            
        case 'gradient'
            
            % Subtly different from the Laplacian. A point for future
            
            % enhancement is to do it better for the triangle interpolation
            
            % case.
            
            [i,j] = meshgrid(1:nx,2:(ny-1));
            
            ind = j(:) + ny*(i(:)-1);
            
            dy1 = dy(j(:)-1)/params.yscale;
            
            dy2 = dy(j(:))/params.yscale;
            
            
            
            Areg = sparse(repmat(ind,1,3),[ind-1,ind,ind+1], ...
                xyRelativeStiffness(2)*[-2./(dy1.*(dy1+dy2)), ...
                2./(dy1.*dy2), -2./(dy2.*(dy1+dy2))],ngrid,ngrid);
            
            
            
            [i,j] = meshgrid(2:(nx-1),1:ny);
            
            ind = j(:) + ny*(i(:)-1);
            
            dx1 = dx(i(:)-1)/params.xscale;
            
            dx2 = dx(i(:))/params.xscale;
            
            
            
            Areg = [Areg;sparse(repmat(ind,1,3),[ind-ny,ind,ind+ny], ...
                xyRelativeStiffness(1)*[-2./(dx1.*(dx1+dx2)), ...
                2./(dx1.*dx2), -2./(dx2.*(dx1+dx2))],ngrid,ngrid)];
            
            
            
    end
    
    nreg = size(Areg,1);
    
    
    
    % Append the regularizer to the interpolation equations,
    
    % scaling the problem first. Use the 1-norm for speed.
    
    NA = norm(A,1);
    
    NR = norm(Areg,1);
    
    A = [A;Areg*(smoothparam*NA/NR)];
    
    rhs = [rhs;zeros(nreg,1)];
    
    % do we have a mask to apply?
    
    if params.maskflag
        
        unmasked = find(params.mask);
        
    end
    
    % solve the full system, with regularizer attached
    
    switch params.solver
        
        case {'\' 'backslash'}
            
            if params.maskflag
                
                % there is a mask to use
                
                zgrid=nan(ny,nx);
                
                zgrid(unmasked) = A(:,unmasked)\rhs;
                
            else
                
                % no mask
                
                zgrid = reshape(A\rhs,ny,nx);
                
            end
            
            
            
        case 'normal'
            
            % The normal equations, solved with \. Can be faster
            
            % for huge numbers of data points, but reasonably
            
            % sized grids. The regularizer makes A well conditioned
            
            % so the normal equations are not a terribly bad thing
            
            % here.
            
            if params.maskflag
                
                % there is a mask to use
                
                Aunmasked = A(:,unmasked);
                
                zgrid=nan(ny,nx);
                
                zgrid(unmasked) = (Aunmasked'*Aunmasked)\(Aunmasked'*rhs);
                
            else
                
                zgrid = reshape((A'*A)\(A'*rhs),ny,nx);
                
            end
            
            
            
        case 'symmlq'
            
            % iterative solver - symmlq - requires a symmetric matrix,
            
            % so use it to solve the normal equations. No preconditioner.
            
            tol = abs(max(z)-min(z))*1.e-13;
            
            if params.maskflag
                
                % there is a mask to use
                
                zgrid=nan(ny,nx);
                
                [zgrid(unmasked),flag] = symmlq(A(:,unmasked)'*A(:,unmasked), ...
                    A(:,unmasked)'*rhs,tol,params.maxiter);
                
            else
                
                [zgrid,flag] = symmlq(A'*A,A'*rhs,tol,params.maxiter);
                
                zgrid = reshape(zgrid,ny,nx);
                
            end
            
            % display a warning if convergence problems
            
            switch flag
                
                case 0
                    
                    % no problems with convergence
                case 1
                    % SYMMLQ iterated MAXIT times but did not converge.
                    warning('GRIDFIT:solver',['Symmlq performed ',num2str(params.maxiter), ...
                        ' iterations but did not converge.'])
                    
                case 3
                    % SYMMLQ stagnated, successive iterates were the same
                    warning('GRIDFIT:solver','Symmlq stagnated without apparent convergence.')
                    
                otherwise
                    warning('GRIDFIT:solver',['One of the scalar quantities calculated in',...
                        ' symmlq was too small or too large to continue computing.'])
            end
            
            
            
        case 'lsqr'
            % iterative solver - lsqr. No preconditioner here.
            
            tol = abs(max(z)-min(z))*1.e-13;
            
            if params.maskflag
                % there is a mask to use
                zgrid=nan(ny,nx);
                [zgrid(unmasked),flag] = lsqr(A(:,unmasked),rhs,tol,params.maxiter);
            else
                [zgrid,flag] = lsqr(A,rhs,tol,params.maxiter);
                zgrid = reshape(zgrid,ny,nx);
            end
            
            % display a warning if convergence problems
            
            switch flag
                case 0
                    % no problems with convergence
                case 1
                    % lsqr iterated MAXIT times but did not converge.
                    warning('GRIDFIT:solver',['Lsqr performed ', ...
                        num2str(params.maxiter),' iterations but did not converge.'])
                case 3
                    % lsqr stagnated, successive iterates were the same
                    warning('GRIDFIT:solver','Lsqr stagnated without apparent convergence.')
                case 4
                    warning('GRIDFIT:solver',['One of the scalar quantities calculated in',...
                        ' LSQR was too small or too large to continue computing.'])
            end
    end  % switch params.solver
end  % if params.tilesize...

% only generate xgrid and ygrid if requested.

if nargout>1
    [xgrid,ygrid]=meshgrid(xnodes,ynodes);
end

% ============================================

% End of main function - gridfit

% ============================================

% ============================================

% subfunction - parse_pv_pairs

% ============================================

function params=parse_pv_pairs(params,pv_pairs)

% parse_pv_pairs: parses sets of property value pairs, allows defaults

% usage: params=parse_pv_pairs(default_params,pv_pairs)

%

% arguments: (input)

%  default_params - structure, with one field for every potential

%             property/value pair. Each field will contain the default

%             value for that property. If no default is supplied for a

%             given property, then that field must be empty.

%

%  pv_array - cell array of property/value pairs.

%             Case is ignored when comparing properties to the list

%             of field names. Also, any unambiguous shortening of a

%             field/property name is allowed.

%

% arguments: (output)

%  params   - parameter struct that reflects any updated property/value

%             pairs in the pv_array.

%

% Example usage:

% First, set default values for the parameters. Assume we

% have four parameters that we wish to use optionally in

% the function examplefun.

%

%  - 'viscosity', which will have a default value of 1

%  - 'volume', which will default to 1

%  - 'pie' - which will have default value 3.141592653589793

%  - 'description' - a text field, left empty by default

%

% The first argument to examplefun is one which will always be

% supplied.

%

%   function examplefun(dummyarg1,varargin)

%   params.Viscosity = 1;

%   params.Volume = 1;

%   params.Pie = 3.141592653589793

%

%   params.Description = '';

%   params=parse_pv_pairs(params,varargin);

%   params

%

% Use examplefun, overriding the defaults for 'pie', 'viscosity'

% and 'description'. The 'volume' parameter is left at its default.

%

%   examplefun(rand(10),'vis',10,'pie',3,'Description','Hello world')

%

% params =

%     Viscosity: 10

%        Volume: 1

%           Pie: 3

%   Description: 'Hello world'

%

% Note that capitalization was ignored, and the property 'viscosity'

% was truncated as supplied. Also note that the order the pairs were

% supplied was arbitrary.



npv = length(pv_pairs);

n = npv/2;



if n~=floor(n)
    
    error 'Property/value pairs must come in PAIRS.'
    
end

if n<=0
    
    % just return the defaults
    
    return
    
end



if ~isstruct(params)
    
    error 'No structure for defaults was supplied'
    
end



% there was at least one pv pair. process any supplied

propnames = fieldnames(params);

lpropnames = lower(propnames);

for i=1:n
    
    p_i = lower(pv_pairs{2*i-1});
    
    v_i = pv_pairs{2*i};
    
    
    
    ind = strmatch(p_i,lpropnames,'exact');
    
    if isempty(ind)
        
        ind = find(strncmp(p_i,lpropnames,length(p_i)));
        
        if isempty(ind)
            
            error(['No matching property found for: ',pv_pairs{2*i-1}])
            
        elseif length(ind)>1
            
            error(['Ambiguous property name: ',pv_pairs{2*i-1}])
            
        end
        
    end
    
    p_i = propnames{ind};
    
    
    
    % override the corresponding default in params
    
    params = setfield(params,p_i,v_i); %#ok
    
    
    
end





% ============================================

% subfunction - check_params

% ============================================

function params = check_params(params)



% check the parameters for acceptability

% smoothness == 1 by default

if isempty(params.smoothness)
    
    params.smoothness = 1;
    
else
    
    if (numel(params.smoothness)>2) || any(params.smoothness<=0)
        
        error 'Smoothness must be scalar (or length 2 vector), real, finite, and positive.'
        
    end
    
end



% regularizer  - must be one of 4 options - the second and

% third are actually synonyms.

valid = {'springs', 'diffusion', 'laplacian', 'gradient'};

if isempty(params.regularizer)
    
    params.regularizer = 'diffusion';
    
end

ind = find(strncmpi(params.regularizer,valid,length(params.regularizer)));

if (length(ind)==1)
    
    params.regularizer = valid{ind};
    
else
    
    error(['Invalid regularization method: ',params.regularizer])
    
end



% interp must be one of:

%    'bilinear', 'nearest', or 'triangle'

% but accept any shortening thereof.

valid = {'bilinear', 'nearest', 'triangle'};

if isempty(params.interp)
    
    params.interp = 'triangle';
    
end

ind = find(strncmpi(params.interp,valid,length(params.interp)));

if (length(ind)==1)
    
    params.interp = valid{ind};
    
else
    
    error(['Invalid interpolation method: ',params.interp])
    
end



% solver must be one of:

%    'backslash', '\', 'symmlq', 'lsqr', or 'normal'

% but accept any shortening thereof.

valid = {'backslash', '\', 'symmlq', 'lsqr', 'normal'};

if isempty(params.solver)
    
    params.solver = '\';
    
end

ind = find(strncmpi(params.solver,valid,length(params.solver)));

if (length(ind)==1)
    
    params.solver = valid{ind};
    
else
    
    error(['Invalid solver option: ',params.solver])
    
end



% extend must be one of:

%    'never', 'warning', 'always'

% but accept any shortening thereof.

valid = {'never', 'warning', 'always'};

if isempty(params.extend)
    
    params.extend = 'warning';
    
end

ind = find(strncmpi(params.extend,valid,length(params.extend)));

if (length(ind)==1)
    
    params.extend = valid{ind};
    
else
    
    error(['Invalid extend option: ',params.extend])
    
end



% tilesize == inf by default

if isempty(params.tilesize)
    
    params.tilesize = inf;
    
elseif (length(params.tilesize)>1) || (params.tilesize<3)
    
    error 'Tilesize must be scalar and > 0.'
    
end



% overlap == 0.20 by default

if isempty(params.overlap)
    
    params.overlap = 0.20;
    
elseif (length(params.overlap)>1) || (params.overlap<0) || (params.overlap>0.5)
    
    error 'Overlap must be scalar and 0 < overlap < 1.'
    
end



% ============================================

% subfunction - tiled_gridfit

% ============================================

function zgrid=tiled_gridfit(x,y,z,xnodes,ynodes,params)

% tiled_gridfit: a tiled version of gridfit, continuous across tile boundaries

% usage: [zgrid,xgrid,ygrid]=tiled_gridfit(x,y,z,xnodes,ynodes,params)

%

% Tiled_gridfit is used when the total grid is far too large

% to model using a single call to gridfit. While gridfit may take

% only a second or so to build a 100x100 grid, a 2000x2000 grid

% will probably not run at all due to memory problems.

%

% Tiles in the grid with insufficient data (<4 points) will be

% filled with NaNs. Avoid use of too small tiles, especially

% if your data has holes in it that may encompass an entire tile.

%

% A mask may also be applied, in which case tiled_gridfit will

% subdivide the mask into tiles. Note that any boolean mask

% provided is assumed to be the size of the complete grid.

%

% Tiled_gridfit may not be fast on huge grids, but it should run

% as long as you use a reasonable tilesize. 8-)



% Note that we have already verified all parameters in check_params



% Matrix elements in a square tile

tilesize = params.tilesize;

% Size of overlap in terms of matrix elements. Overlaps

% of purely zero cause problems, so force at least two

% elements to overlap.

overlap = max(2,floor(tilesize*params.overlap));



% reset the tilesize for each particular tile to be inf, so

% we will never see a recursive call to tiled_gridfit

Tparams = params;

Tparams.tilesize = inf;



nx = length(xnodes);

ny = length(ynodes);

zgrid = zeros(ny,nx);



% linear ramp for the bilinear interpolation

rampfun = inline('(t-t(1))/(t(end)-t(1))','t');



% loop over each tile in the grid

h = waitbar(0,'Relax and have a cup of JAVA. Its my treat.');

warncount = 0;

xtind = 1:min(nx,tilesize);

while ~isempty(xtind) && (xtind(1)<=nx)
    
    
    
    xinterp = ones(1,length(xtind));
    
    if (xtind(1) ~= 1)
        
        xinterp(1:overlap) = rampfun(xnodes(xtind(1:overlap)));
        
    end
    
    if (xtind(end) ~= nx)
        
        xinterp((end-overlap+1):end) = 1-rampfun(xnodes(xtind((end-overlap+1):end)));
        
    end
    
    
    
    ytind = 1:min(ny,tilesize);
    
    while ~isempty(ytind) && (ytind(1)<=ny)
        
        % update the waitbar
        
        waitbar((xtind(end)-tilesize)/nx + tilesize*ytind(end)/ny/nx)
        
        
        
        yinterp = ones(length(ytind),1);
        
        if (ytind(1) ~= 1)
            
            yinterp(1:overlap) = rampfun(ynodes(ytind(1:overlap)));
            
        end
        
        if (ytind(end) ~= ny)
            
            yinterp((end-overlap+1):end) = 1-rampfun(ynodes(ytind((end-overlap+1):end)));
            
        end
        
        
        
        % was a mask supplied?
        
        if ~isempty(params.mask)
            
            submask = params.mask(ytind,xtind);
            
            Tparams.mask = submask;
            
        end
        
        
        
        % extract data that lies in this grid tile
        
        k = (x>=xnodes(xtind(1))) & (x<=xnodes(xtind(end))) & ...
            (y>=ynodes(ytind(1))) & (y<=ynodes(ytind(end)));
        
        k = find(k);
        
        
        
        if length(k)<4
            
            if warncount == 0
                
                warning('GRIDFIT:tiling','A tile was too underpopulated to model. Filled with NaNs.')
                
            end
            
            warncount = warncount + 1;
            
            
            
            % fill this part of the grid with NaNs
            
            zgrid(ytind,xtind) = NaN;
            
            
            
        else
            
            % build this tile
            
            zgtile = gridfit(x(k),y(k),z(k),xnodes(xtind),ynodes(ytind),Tparams);
            
            
            
            % bilinear interpolation (using an outer product)
            
            interp_coef = yinterp*xinterp;
            
            
            
            % accumulate the tile into the complete grid
            
            zgrid(ytind,xtind) = zgrid(ytind,xtind) + zgtile.*interp_coef;
            
            
            
        end
        
        
        
        % step to the next tile in y
        
        if ytind(end)<ny
            
            ytind = ytind + tilesize - overlap;
            
            % are we within overlap elements of the edge of the grid?
            
            if (ytind(end)+max(3,overlap))>=ny
                
                % extend this tile to the edge
                
                ytind = ytind(1):ny;
                
            end
            
        else
            
            ytind = ny+1;
            
        end
        
        
        
    end % while loop over y
    
    
    
    % step to the next tile in x
    
    if xtind(end)<nx
        
        xtind = xtind + tilesize - overlap;
        
        % are we within overlap elements of the edge of the grid?
        
        if (xtind(end)+max(3,overlap))>=nx
            
            % extend this tile to the edge
            
            xtind = xtind(1):nx;
            
        end
        
    else
        
        xtind = nx+1;
        
    end
    
    
    
end % while loop over x



% close down the waitbar

close(h)



if warncount>0
    
    warning('GRIDFIT:tiling',[num2str(warncount),' tiles were underpopulated & filled with NaNs'])
    
end


