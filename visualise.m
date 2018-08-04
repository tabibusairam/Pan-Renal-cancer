% [filename, pathname] = uigetfile( ...
%     {'*.tif*;*.png;*.jpg;*.svs;*.scn', 'All Image Files (*.tif*, *.png, *.jpg, *.svs, *.scn)';
%     '*.tif*','TIFF images (*.tif, *.tiff)'; ...
%     '*.png','PNG images (*.png)'; ...
%     '*.jpg','JPG images (*.jpg)'; ...
%     '*.svs','SVS images (*.svs)'; ...
%     '*.scn','SCN images (*.scn)'; ...
%     '*.*',  'All Files (*.*)'}, ...
%     'Pick Image');


% cd(pathname)  
% I1info = imfinfo(filename);
% for i=1:numel(I1info)
%     pageinfo1{i}=['Page ' num2str(i) ': ' num2str(I1info(i).Height) ' x ' num2str(I1info(i).Width)];
% end

%%
% fname=[pathname filename];
% 
% 
% if numel(I1info)>1,
%     [s,v]=listdlg('Name','Choose Level','PromptString','Select a page for Roi Discovery:','SelectionMode','single','ListSize',[170 120],'ListString',pageinfo1); drawnow;
%     if ~v, guidata(hObject, handles);return;end
%  io=imread(fname,s);
% end
% 
%  imtool(io)
%  
%  imwrite(io, '/Users/user/Desktop/clinical_data/3.jpg');
% % h=imrect;
% % roi = wait(h);

%%



% path='/Volumes/Sairam/kidney-c';
% path1 = '/Users/sairam/Desktop/kid-svs/mat-files/kidney-c';
% path2 = '/Users/sairam/Desktop/kid-svs/svs-files/kidney-c';
% 
% for i=1:3
%     current_path = strcat(path,num2str(i));
%     disp(current_path)
%     cd(current_path)
%     load(strcat(path1,num2str(i),'/test/','data.mat'));
%     mkdir(strcat(path2,num2str(i)));
%     svs_files = dir('**/*.svs');
%     
%     for j=1:size(files,1)
%         disp(j)
%         for k=1:size(svs_files,1)
%             if(strcmp(files(j).name,svs_files(k).name) == 1)
%                 path_to_svs = strcat(svs_files(k).folder,'/',svs_files(k).name);
%                 copyfile(path_to_svs,strcat(path2,num2str(i)));
%             end
%         end
%     end
%     
% end


%%
% 
% classes = importdata('/Users/sairam/Desktop/kidney/resnet-34/subtype/classes1.txt');
% probabilities = importdata('/Users/sairam/Desktop/kidney/resnet-34/subtype/probabilities1.txt');
% path='/Users/sairam/Desktop/kid-svs/kidney-c';
% path1 = '/Users/sairam/Desktop/kid-svs/mat-files/kidney-c';
% path2 = '/Users/sairam/Desktop/kid-svs/svs-files/kidney-c';
% 
% for class = 1:3
%     
%     load(strcat(path1,num2str(class),'/test/data.mat'))
%     cd(strcat(path2,num2str(class)))
%     svs_files = dir('**/*.svs');
%     
%     disp(strcat(path2,num2str(class)))
%     
%     count = 1;
%     cd(strcat(path,num2str(class),'/mat_files'))
%     mat_files = dir('**/*.mat');
%     
%     mkdir(strcat(path,num2str(class),'/images1'))
%     
%     if(class == 2)
%         pointer = 42907;
%     elseif(class == 3)
%         pointer = 42907 + 13577;
%     else
%         pointer = 0;
%     end
%     
%     prob = 1.0 ./ ( 1.0 + exp(-probabilities));
%     a=prob>0.9;
%     prob = a.*prob;
%     
%     for i = 1:size(mat_files,1)
%         disp(i)
%         gi = str2num(mat_files(i).name(1:end-4));
%         for fi = 1:size(svs_files,1)
%             if(strcmp(files(gi).name,svs_files(fi).name) == 1)
%                 image = imread(strcat(svs_files(fi).folder,'/',svs_files(fi).name),3);
%                 imshow(image)
%             end
%         end
%         
%         
%         
%         load(strcat(mat_files(i).folder,'/',mat_files(i).name))
%         for j=1:size(scores,1)
%             if(scores(j,3) == 1)
%                 
%                 if(classes(count + pointer) == 0)
%                     clr = 'g';
%                 elseif(classes(count + pointer) == 1)
%                     clr = 'b';
%                 else
%                     clr = 'r';
%                 end
%                 
%                 transp = prob(count + pointer) * 0.3;
%                 x=[scores(j,1) scores(j,1)+511 scores(j,1)+511 scores(j,1)];
%                 y=[scores(j,2) scores(j,2) scores(j,2)+511 scores(j,2)+511];
%                 
%                 patch(y,x,clr,'EdgeColor',clr,'FaceAlpha',transp)
%                 
%                 count = count + 1;
%                 
%             end
%         end
%         
%         img = getframe(gcf);
%         imwrite(img.cdata, strcat(path,num2str(class),'/images1/',num2str(gi),'.png'));
%     end
%     
%     
% 
%     
% end

%%

% classes = importdata('/Users/sairam/Desktop/kidney/resnet-34/cancer_normal/class3/classes1.txt');
% probabilities = importdata('/Users/sairam/Desktop/kidney/resnet-34/cancer_normal/class3/probabilities1.txt');
% path='/Users/sairam/Desktop/kid-svs/kidney-c';
% path1 = '/Users/sairam/Desktop/kid-svs/mat-files/kidney-c';
% path2 = '/Users/sairam/Desktop/kid-svs/svs-files/kidney-c';
% 
% %class 1 - 24301 and 42907
% %class 2 - 32538 and 45218
% 
% 
% class = 3;
%     
%     load(strcat(path1,num2str(class),'/test/data.mat'))
%     cd(strcat(path2,num2str(class)))
%     svs_files = dir('**/*.svs');
%     
%     disp(strcat(path2,num2str(class)))
%     
%     count = 1;
%     cd(strcat(path,num2str(class),'/mat_files'))
%     mat_files = dir('**/*.mat');
%     
%     mkdir(strcat(path,num2str(class),'/cancer_normal-images1'))
%     
%     if(class == 1)
%         pointer = 24301;
%     elseif(class == 3)
%         pointer = 32538;
%     end
%     
%     prob = 1.0 ./ ( 1.0 + exp(-probabilities));
%     a=prob>0.85;
%     prob = a.*prob;
%     
%     for i = 1:size(mat_files,1)
%         disp(i)
%         gi = str2num(mat_files(i).name(1:end-4));
%         for fi = 1:size(svs_files,1)
%             if(strcmp(files(gi).name,svs_files(fi).name) == 1)
%                 image = imread(strcat(svs_files(fi).folder,'/',svs_files(fi).name),3);
%                 imshow(image)
%             end
%         end
%         
%         
%         
%         load(strcat(mat_files(i).folder,'/',mat_files(i).name))
%         for j=1:size(scores,1)
%             if(scores(j,3) == 1)
%                 
%                 if(classes(count + pointer) == 0)
%                     clr = 'r';
%                 elseif(classes(count + pointer) == 1)
%                     clr = 'g';
%                 end
%                 
%                 transp = prob(count + pointer) * 0.3;
%                 x=[scores(j,1) scores(j,1)+511 scores(j,1)+511 scores(j,1)];
%                 y=[scores(j,2) scores(j,2) scores(j,2)+511 scores(j,2)+511];
%                 
%                 patch(y,x,clr,'EdgeColor',clr,'FaceAlpha',transp)
%                 
%                 count = count + 1;
%                 
%             end
%         end
%         
%         img = getframe(gcf);
%         imwrite(img.cdata, strcat(path,num2str(class),'/cancer_normal-images1/',num2str(gi),'.png'));
%     end
%     
%     

    

%%





classes = importdata('/Users/sairam/Desktop/kidney/resnet-34/subtype/classes1.txt');
classes1 = importdata('/Users/sairam/Desktop/kidney/resnet-34/cancer_normal/class3/classes1.txt');

probabilities = importdata('/Users/sairam/Desktop/kidney/resnet-34/subtype/probabilities1.txt');
probabilities1 = importdata('/Users/sairam/Desktop/kidney/resnet-34/cancer_normal/class3/probabilities1.txt');
path='/Users/sairam/Desktop/kid-svs/kidney-c';
path1 = '/Users/sairam/Desktop/kid-svs/mat-files/kidney-c';
path2 = '/Users/sairam/Desktop/kid-svs/svs-files/kidney-c';

%class 1 - 24301 and 42907
%class 2 - 32538 and 45218


case_ids = [1 ; 3 ];

class=3;
    
    load(strcat(path1,num2str(class),'/test/data.mat'))
    cd(strcat(path2,num2str(class)))
    svs_files = dir('**/*.svs');
    
    disp(strcat(path2,num2str(class)))
    
    count = 1;
    cd(strcat(path,num2str(class),'/mat_files'))
    mat_files = dir('**/*.mat');
    
    mkdir(strcat(path,num2str(class),'/texture-images1'))
    
    if(class == 1)
        pointer = 0;
    elseif(class == 3)
        pointer = 42907 + 13577;
    end
    
        
    if(class == 1)
        pointer1 = 24301;
    elseif(class == 3)
        pointer1 = 32538;
    end
    
    
    
    prob = 1.0 ./ ( 1.0 + exp(-probabilities));
    a=prob>0.9;
    prob = a.*prob;
    
    prob1 = 1.0 ./ ( 1.0 + exp(-probabilities1));
    prob1=prob1>0.85;
    
    for i = 1:size(mat_files,1)
        disp(i)
        gi = str2num(mat_files(i).name(1:end-4));
        for fi = 1:size(svs_files,1)
            if(strcmp(files(gi).name,svs_files(fi).name) == 1)
                image = imread(strcat(svs_files(fi).folder,'/',svs_files(fi).name),3);
                imshow(image)
            end
        end
        
        
        
        load(strcat(mat_files(i).folder,'/',mat_files(i).name))
        for j=1:size(scores,1)
            if(scores(j,3) == 1)
                                
                transp = prob(count + pointer) * 0.3;
                x=[scores(j,1) scores(j,1)+511 scores(j,1)+511 scores(j,1)];
                y=[scores(j,2) scores(j,2) scores(j,2)+511 scores(j,2)+511];
                
                if(classes(count + pointer) == 2)
                    if(prob1(count+pointer1) == 0)
                        clr = 'r';
                        patch(y,x,clr,'EdgeColor',clr,'FaceAlpha',transp)
                    end
%                 elseif(classes(count + pointer) == 1)
%                     if(prob1(count+pointer1) == 0)
%                         clr = 'g';
%                         patch(y,x,clr,'EdgeColor',clr,'FaceAlpha',transp)
%                     end
                end
                
                
                count = count + 1;
                
            end
        end
        
        img = getframe(gcf);
        imwrite(img.cdata, strcat(path,num2str(class),'/texture-images1/',num2str(gi),'.png'));
    end
    







