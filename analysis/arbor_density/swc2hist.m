% Generate arbor densities from neuron reconstruction in swc format for
% list of specimens specified by csv file
rootdir = "root_dir";
indir = strcat(rootdir, "indir"); % pia/wm csv
swcdir = strcat(rootdir, "swcdir"); % traces
outdir = strcat(rootdir, "output_files");
sp_list_file = strcat(rootdir, "specimens.csv");

% get specimens id
fdi = fopen(sp_list_file);
out = textscan(fdi, '%s', 'delimiter', ',','headerLines',1);
fclose(fdi);
specimens = out{1};
% setup parameters
swcUnitType = [-1 0 1 2 3 4 5];
layers_ext= '.csv';
Borders_all = {'pia', 'layer23', 'layer4', 'layer5', 'layer6a', 'layer6b', 'wm'};
line_all_ext = strcat('_upright_', Borders_all, layers_ext);
Borders_piawm = {'pia', 'wm'};
line_piawm_ext = strcat('_upright_', Borders_piawm, layers_ext);
voxelRes = [1,1,1];
conformalJump = 10;
avg_layer_thickness = ([1, 115, 333, 454, 688, 883, 923]-1)/(923-1);
units_swc = "pixels";
nHeader = 0;
scalefactor = 1;
extstr = '';

for i=1:length(specimens)
    sp = specimens{i};
    disp(['sp ', sp])
    arborFilename = strcat(swcdir, sp, extstr, '_upright.swc');
    surfaceFilenames_all = strcat(indir, sp, line_all_ext);
    surfaceFilenames_piawm = strcat(indir, sp, line_piawm_ext);
    rgcAnalyzer(outdir, arborFilename, swcUnitType, surfaceFilenames_piawm, voxelRes, conformalJump, avg_layer_thickness, ...
                Borders_piawm, Borders_all, units_swc, nHeader, scalefactor, sp, extstr);
end

