T_file = '';
soma_file = '';
inverted_tif_dir = '';
save_dir = '';

T_all(sx_min:sx_max,sy_min:sy_max,sz_min:sz_max) = max(thresh_soma,T_all(sx_min:sx_max,sy_min:sy_max,sz_min:sz_max));
T_all(isinf(T_all)) = 0;
T_all(T_all>0) = 1;
T_all = uint8(T_all);

s = size(T_all);
x_crop = floor(s(1)/64)*64;
y_crop = floor(s(2)/64)*64;
z_crop = floor(s(3)/10)*10;

T_all = T_all(1:x_crop,1:y_crop,1:z_crop);
save([save_dir, 'labels.mat'], '-v7.3', 'T_all');
mat2tifslice(T_all,save_dir+'labels/')

fname = {};
fname = {};
[Orig,~,~] = ImageStackJ([fname,'']);
Orig = Orig(1:x_crop,1:y_crop,1:z_crop);
mat2tifslice(Orig,save_dir+'inputs/')