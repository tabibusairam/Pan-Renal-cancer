% [filename, pathname] = uigetfile( ...
%     {'*.tif*;*.png;*.jpg;*.svs;*.scn', 'All Image Files (*.tif*, *.png, *.jpg, *.svs, *.scn)';
%     '*.tif*','TIFF images (*.tif, *.tiff)'; ...
%     '*.png','PNG images (*.png)'; ...
%     '*.jpg','JPG images (*.jpg)'; ...
%     '*.svs','SVS images (*.svs)'; ...
%     '*.scn','SCN images (*.scn)'; ...
%     '*.*',  'All Files (*.*)'}, ...
%     'Pick Image');
% 
% imginfo = [pathname filename];
% cd(pathname)
% I1info = imfinfo(filename);
% img=imread(imginfo,3);
% R = double(img(:,:,1));
% G = double(img(:,:,2));
% B = double(img(:,:,3));
% blueRatio = uint8(((100 * B)./(1+R+G)) .* (256./(1+B+R+G)));
% dim=[256 256];
% dim1=256;
% [img_x,img_y] = size(blueRatio);
% scores=[];
% 
% SE = strel('square',2);
% a=imdilate(blueRatio,SE);
% k=1;
% i=1;
% while(i<=50000)
%     x= ceil(randi(img_x-dim1));
%     y= ceil(randi(img_y-dim1));
%     im = blueRatio(x:x+dim1,y:y+dim1);
%     addi = sum(sum(im));
%     scores(k,:)= [x y addi];
%     k=k+1;
%     i=i+1;
%     
% end
% 
% final = sortrows(scores,3,'descend');




%%

%
%
mode = ['colon-valid'];
%classes = ['above8','below7']
%path = '/Volumes/Sairam/colorectal-svs/colon';
path = '/Volumes/Sairam/colorectal-svs/colon';
path2 = '/Users/user/Desktop/colorectal/colon/colon-valid/';

colon_valid = [9;25;31;32;34;66;72;76;82;84;87;102;103];

for m=1:size(mode,1)
%    for i=1:size(classes,1)
        %if (strcmp(classes(i,:),'above8') && strcmp(mode(m,:),'train')),
        %    continue
        %end
%        disp(classes(i,:))
        disp(mode(m,:))
        path1 = strcat(path,'/',mode(m,:));
        cd(path1);
        %files = dir('**/*.svs');
       %[~,index] = sortrows({files.folder}.'); files = files(index); clear index
 
        load('data.mat');
       %save('data.mat','files');

        if(m==1)
            img_fn = 1;
        else
            img_fn = 0;
        end
  
        for j=1:size(colon_valid,1)
            %size(files,1)
            disp(j)
            disp(colon_valid(j)+1)
            %disp('hi')
            %if(j~=122)
            %   continue;
            %end
            image = imread(strcat(files(colon_valid(j)+1).folder,'/',files(colon_valid(j)+1).name),1);
            %disp('hi1')
            box_info = boxes_1(image,img_fn);
            %tag1 = strcat(num2str(train(j+1)),'.mat');
            %save(tag1,'box_info');
            for k=1:size(box_info,1)
                patch = image(box_info(k,1):box_info(k,1)+256,box_info(k,2):box_info(k,2)+256,:);
                tag = strcat(path2, num2str(colon_valid(j)+1),'-',num2str(k),'.png');
                imwrite(patch, tag, 'mode', 'lossless');
            end
        
        end
end     
    
    
            









