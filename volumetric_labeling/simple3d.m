%--------------------------------------------------------------------------
function s=simple3d(im,n)
% Decides whether the central point of IM is a simple point
% 6-simple means that T_6(IM)=1 and T_26(~IM)=1 -- 26-simple is the opposite
% authors: Viren Jain, Mark Richardson, M.I.T.

if ndims(im) ~=3
    error('image patch must be 3d')
end
if any(size(im)~=[3 3 3])
    error('must be a 3x3x3 image patch')
end

switch n
    case 6
        if topo(im,6)==1 && topo(1-im,26)==1
            s=1;
        else
            s=0;
        end
    case 26
        if topo(im,26)==1 && topo(1-im,6)==1
            s=1;
        else
            s=0;
        end
    otherwise
        error('n must be 4 or 8')
end

%--------------------------------------------------------------------------
function t=topo(im,n)
% Computes topological numbers for the central point of an image patch.
% These numbers can be used as the basis of a topological classification.
% T_4 and T_8 are used when IM is a 2d image patch of size 3x3
% T_6 and T_26 are used when IM is a 3d image patch of size 3x3x3,
% defined on p. 172 of Bertrand & Malandain, Patt. Recog. Lett. 15, 169-75 (1994).
% authors: Viren Jain, Mark Richardson, M.I.T.

switch n
    case 4
        % number of 4-connected components in the 8-neighborhood of the
        % center that are 4-adjacent to the center
        if ndims(im) ~= 2
            error('n=4 is valid for a 2d image')
        end
        if any(size(im)~=[3 3])
            error('must be 3x3 image patch')
        end
        neighbor4=[0 1 0; 1 0 1; 0 1 0];
        im(2,2)=0;    % ignore the central point
        components=bwlabel(im,4).*neighbor4;  % zero out locations that are not in the four-neighborhood
    case 8
        % number of 8-connected components in the 8-neighborhood of the
        % center (adjacency is automatic)
        if ndims(im) ~= 2
            error('n=8 is valid for a 2d image')
        end
        if any(size(im)~=[3 3])
            error('must be 3x3 image patch')
        end
        im(2,2)=0;  % ignore the central point
        components=bwlabel(im,8);
    case 6
        % number of 6-connected components in the 18-neighborhood of the center
        % that are 6-adjacent to the center
        if ndims(im) ~= 3
            error('n=6 is valid for a 3d image')
        end
        if any(size(im)~=[3 3 3])
            error('must be 3x3x3 image patch')
        end
        neighbor6=conndef(3,'min');  % the nearest six neighbors
        neighbor18=ones(3,3,3); neighbor18(1:2:3,1:2:3,1:2:3)=0; neighbor18(2,2,2)=0;   % the nearest 18 neighbors
        components=bwlabeln(neighbor18.*im,6);  % 6-connected components in the 18 neighborhood of the center
        components=components.*neighbor6;  % keep components that are 6-adjacent to the center
    case 26
        % number of 26-components in the 26-neighborhood of the center
        % (adjacency is automatic)
        if ndims(im) ~= 3
            error('n=26 is valid for a 3d image')
        end
        if any(size(im)~=[3 3 3])
            error('must be 3x3x3 image patch')
        end
        im(2,2,2)=0;
        components=bwlabeln(im,26);
    otherwise
        error('n must be 4, 8, 6, or 26')
end
t=length(unique(nonzeros(components)));
