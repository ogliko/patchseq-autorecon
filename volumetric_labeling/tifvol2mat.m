function IM = tifvol2mat(fname)
InfoImage=imfinfo(fname);
mImage=InfoImage(1).Width;
nImage=InfoImage(1).Height;
NumberImages=length(InfoImage);
IM=zeros(nImage,mImage,NumberImages,'double');

TifLink = Tiff(fname, 'r');
for t=1:NumberImages
    TifLink.setDirectory(t);
    IM(:,:,t)=TifLink.read();
end
TifLink.close();
end