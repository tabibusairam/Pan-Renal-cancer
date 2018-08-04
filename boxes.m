function [ final_boxes ] = boxes( img , img_fn )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
R = (img(:,:,1));
G = (img(:,:,2));
B = (img(:,:,3));
[img_x,img_y] = size(R);
dim1=256;
i=1;
k=1;
scores=[];
if(img_fn == 1)
    num_box = 2000;
else
    num_box = 2000;
end

while(i<=25000)
    x= ceil(randi(img_x-dim1));
    y= ceil(randi(img_y-dim1));    
    R_d = double(R(x:x+dim1,y:y+dim1));
    G_d = double(G(x:x+dim1,y:y+dim1));
    B_d = double(B(x:x+dim1,y:y+dim1));
    blueRatio = uint8(((100 * B_d)./(1+R_d+G_d)) .* (256./(1+B_d+R_d+G_d)));
    
%     SE = strel('square',2);
%     a=imdilate(blueRatio,SE);
    
    addi = sum(sum(blueRatio));
    scores(k,:)= [x y addi];
    k=k+1;
    i=i+1;
    
end

final = sortrows(scores,3,'descend');
final_boxes = final(1:num_box,:);

end

