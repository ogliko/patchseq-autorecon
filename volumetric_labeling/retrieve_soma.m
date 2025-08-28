function [thresh_soma, soma_patch, r, sx_min,sx_max,sy_min,sy_max,sz_min,sz_max] = retrieve_soma(r,R,t,MaxDist,im_fileid,start_im_offset,threshold)
    soma_r = r(t==1, :);
    soma_R = R(t==1) - (MaxDist*3);

    soma_R_grid = round(soma_R);
    [dx, dy, dz] = meshgrid(-soma_R_grid:soma_R_grid, -soma_R_grid:soma_R_grid, -soma_R_grid:soma_R_grid);
    temp =  [dx(:), dy(:), dz(:)];
    temp=temp(dx(:).^2+dy(:).^2+(dz(:)*3).^2<soma_R.^2,:);
    temp = bsxfun(@plus,temp,soma_r);
    r = [r;temp];

    sx_min = round(min(temp(:,1)))-50;
    sx_max = round(max(temp(:,1)))+50;
    sy_min = round(min(temp(:,2)))-50;
    sy_max = round(max(temp(:,2)))+50;
    sz_min = max(round(min(temp(:,3)))-50,1);
    sz_max = round(max(temp(:,3)))+50;
    
    [soma_patch] = fetch_patch(sx_min,sx_max,sy_min,sy_max,sz_min,sz_max,im_fileid,start_im_offset);

    SE = strel('sphere',5);
    eroded_soma = imerode(soma_patch, SE);
    eroded_soma = imerode(eroded_soma, SE);
    eroded_soma = imerode(eroded_soma, SE);
    dilated_soma = imdilate(eroded_soma, SE);
    dilated_soma = imdilate(dilated_soma, SE);
    thresh_soma = dilated_soma;
    thresh_soma(dilated_soma<threshold) = 0;
    thresh_soma(thresh_soma>0) = 1;
end

