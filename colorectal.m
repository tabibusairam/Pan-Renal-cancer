img=imread('colon-25.png');

R_d = double(img(:,:,1));
G_d = double(img(:,:,2));
B_d = double(img(:,:,3));

blueRatio = uint8(((100 * B_d)./(1+R_d+G_d)) .* (256./(1+B_d+R_d+G_d)));
imtool(blueRatio)
s = regionprops(blueRatio,'BoundingBox')
