% phase = 'test';
% 
% 
% load(strcat('/Users/sairam/Desktop/kid-svs/mat-files/kidney-c3/', phase, '/data.mat'));
% 
% 
% path_svs = (strcat('/Volumes/Sairam/kidney-c3/', phase , '-c3'));
% 
% classes = importdata(strcat('/Users/sairam/Desktop/kidney/cancer_normal_40x/class3/resnet-18/',phase,'/classes.txt'));
% probability = importdata(strcat('/Users/sairam/Desktop/kidney/cancer_normal_40x/class3/resnet-18/',phase,'/probabilities.txt'));
% 
% slide_info=importdata(strcat('/Users/sairam/Desktop/kid-svs/svs-files/kidney-c3/',phase,'.txt'));
% 
% final_path = '/Users/sairam/Desktop/survival_experiments-c3/';
% prob = 1.0 ./ ( 1.0 + exp(-probability));
% 
% if(strcmp(phase,'test')==1)
%     index = 126294;
% elseif(strcmp(phase,'valid') == 1)
%     index = 78663;
% elseif(strcmp(phase,'train') == 1)
%     index = 374812;
% end
% 
% 
% 
% 
% 
% cd(path_svs)
% count = 0;
% counter = 0;
% svs_files = dir('**/*.svs');
% 
% for sld = 1:size(files,1)
%     disp(sld)
%     disp(files(sld).name(1:12))
%     if(slide_info(sld) == 0)
%         continue
%     end
%     load(strcat('/Users/sairam/Desktop/kid-svs/test-40x/mat-files-c3/', phase , '-c3-mat-files/' , num2str(sld) ,'.mat'));
%     
%     
% %    img = imread(files(sld).name,1);
%     for gi = 1:size(svs_files,1)
%         if(strcmp(files(sld).name , svs_files(gi).name) == 1)
%             I1info = imfinfo(files(sld).name);
% 
%         end
%     end
% 
% 
%     binary_img = zeros(I1info(1).Height,I1info(1).Width);
%     
%     for i = 1:size(scores,1)
%         if(scores(i,3) == 1)
%             count = count + 1;
%             if(prob(index+count) > 0.95)
%                 if(classes(index+count)  == 1)
%                     binary_img(scores(i,1): scores(i,1) + 511, scores(i,2): scores(i,2) + 511) = binary_img(scores(i,1): scores(i,1) + 511, scores(i,2): scores(i,2) + 511) + 255;
%                     counter = counter + 1;
%                 end
%  %               disp(counter)
%             end
%         
%         
%         end
%     
%     end
%     saving = strcat(final_path , files(sld).name(1:12) , '/' , phase , '-' , num2str(sld) , '.png' );
%     binary_img = uint8(binary_img);
% 
% %imwrite(img, strcat(saving,'normal.png'), 'mode', 'lossless');
% %      if(sld ~=34)
%         imwrite(binary_img, saving, 'mode', 'lossless');
% %        end
%     disp(files(sld).name(1:12))
% 
% 
% 
% end



%%


binary_path = '/Users/sairam/Desktop/survival_experiments_clean/';
cd(binary_path)
files = dir();
files(1:3) = [];

%cell_path = '/Users/sairam/Desktop/survival_experiments_clean-c3/';


final_clinical(size(files,1)) = struct();

for i = 1:size(files,1)
%     if(i==3)
%         continue
%     end
    disp(i)
    disp(files(i).name)
    
    total_area = 0;
    total_convex_area = 0;
    total_perimeter = 0;
    total_filled_area = 0;
    total_major_axis = 0;
    total_minor_axis = 0;
    total_peri_by_area = 0;
    main_region_area = 0;
    main_region_convex_area = 0;
    main_region_eccentricity = 0;
    main_region_extent = 0;
    main_region_solidity = 0;
    main_region_perimeter = 0;
    main_region_angle = 0;
    main_region_peri_by_area = 0;
    main_region_major_axis = 0;
    main_region_minor_axis = 0;
    
    
    imgs = dir(strcat(binary_path, files(i).name ,  '/*.png'));
    if(size(imgs,1) == 0)
        total_area = 0;
        total_convex_area = 0;
        total_perimeter = 0;
        total_filled_area = 0;
        total_major_axis = 0;
        total_minor_axis = 0;
        total_peri_by_area = 0;
        main_region_area = 0;
        main_region_convex_area = 0;
        main_region_eccentricity = 0;
        main_region_extent = 0;
        main_region_solidity = 0;
        main_region_perimeter = 0;
        main_region_angle = 0;
        main_region_peri_by_area = 0;
        main_region_major_axis = 0;
        main_region_minor_axis = 0;
        
        
    else
        for j = 1:size(imgs,1)
            img = imread(strcat(imgs(j).folder,'/',imgs(j).name));
            CC = bwconncomp(img, 8);
            if(CC.NumObjects > 0)
%                 numPixels = cellfun(@numel,CC.PixelIdxList);
%                 removal = (numPixels > (max(numPixels)/3));
% 
%                 for idx = 1:size(removal,2)
%                     if(removal(idx) == 0)
%                         img(CC.PixelIdxList{idx}) = 0;
%                     end
%                 end
%                 
%                 imwrite(img,strcat(cell_path,files(i).name,'/',imgs(j).name),'mode', 'lossless')
                CC = bwconncomp(img,8);
                numPixels = cellfun(@numel,CC.PixelIdxList);
                [biggest, idx] = max(numPixels);
                stats = regionprops(CC,'all');
                
                % total Area
                for ind = 1:size(stats,1)
                    total_area = total_area + stats(ind).Area;
                end
                %Total Convex Area
                for ind = 1:size(stats,1)
                    total_convex_area = total_convex_area + stats(ind).ConvexArea;
                end
                %total Perimeter
                for ind = 1:size(stats,1)
                    total_perimeter = total_perimeter + stats(ind).Perimeter;
                end                
                %total Filled Area
                for ind = 1:size(stats,1)
                    total_filled_area = total_filled_area + stats(ind).FilledArea;
                end
                %total major axis
                for ind = 1:size(stats,1)
                    total_major_axis = total_major_axis + stats(ind).MajorAxisLength;
                end              
                %total Minor axis
                for ind = 1:size(stats,1)
                    total_minor_axis = total_minor_axis + stats(ind).MinorAxisLength;
                end 
                %total peri^2 by area
                for ind = 1:size(stats,1)
                    total_peri_by_area = total_peri_by_area + (stats(ind).Perimeter^2 / stats(ind).Area);
                end 
                %main region area
                main_region_area = stats(idx).Area;
                %main region convex area
                main_region_convex_area = stats(idx).ConvexArea;                
                %main region eccentricity
                main_region_eccentricity = stats(idx).Eccentricity;                
                %main region extent
                main_region_extent = stats(idx).Extent;
                %main region solidity
                main_region_solidity = stats(idx).Solidity;
                %main region perimeter
                main_region_perimeter = stats(idx).Perimeter;
                %main region angle
                main_region_angle = stats(idx).Orientation;
                %main region peri by area
                main_region_peri_by_area = ((stats(idx).Perimeter)^2 / stats(idx).Area);
                %main region minor axis
                main_region_minor_axis = stats(idx).MinorAxisLength;
                %major axis
                main_region_major_axis = stats(idx).MajorAxisLength;

            else     % for 0 CC objects
                total_area = total_area + 0;
                total_convex_area = total_convex_area+0;
                total_perimeter = total_perimeter+0;
                total_filled_area = total_filled_area+0;
                total_major_axis = total_major_axis+0;
                total_minor_axis = total_minor_axis+0;
                total_peri_by_area = total_peri_by_area+0;
                main_region_area = main_region_area+0;
                main_region_convex_area = main_region_convex_area+0;
                main_region_eccentricity = main_region_eccentricity+0;
                main_region_extent = main_region_extent+0;
                main_region_solidity = main_region_solidity+0;
                main_region_perimeter = main_region_perimeter+0;
                main_region_angle = main_region_angle+0;
                main_region_peri_by_area = main_region_peri_by_area+0;
                main_region_major_axis = main_region_major_axis+0;
                main_region_minor_axis = main_region_minor_axis+ 0;
                
            end

        end
        total_area = total_area/size(imgs,1);
        total_convex_area = total_convex_area/size(imgs,1);
        total_perimeter = total_perimeter/size(imgs,1);
        total_filled_area = total_filled_area/size(imgs,1);
        total_major_axis = total_major_axis/size(imgs,1);
        total_minor_axis = total_minor_axis/size(imgs,1);
        total_peri_by_area = total_peri_by_area/size(imgs,1);
        main_region_area = main_region_area/size(imgs,1);
        main_region_convex_area = main_region_convex_area/size(imgs,1);
        main_region_eccentricity = main_region_eccentricity/size(imgs,1);
        main_region_extent = main_region_extent/size(imgs,1);
        main_region_solidity = main_region_solidity/size(imgs,1);
        main_region_perimeter = main_region_perimeter/size(imgs,1);
        main_region_angle = main_region_angle/size(imgs,1);
        main_region_peri_by_area = main_region_peri_by_area/size(imgs,1);
        main_region_major_axis = main_region_major_axis/size(imgs,1);
        main_region_minor_axis = main_region_minor_axis/size(imgs,1);             
    end
    
    final_clinical(i).submitter_id = files(i).name;
    final_clinical(i).total_area = total_area;
    final_clinical(i).total_convex_area = total_convex_area;
    final_clinical(i).total_perimeter = total_perimeter;
    final_clinical(i).total_filled_area = total_filled_area;
    final_clinical(i).total_major_axis = total_major_axis;
    final_clinical(i).total_minor_axis = total_minor_axis;
    final_clinical(i).total_peri_by_area = total_peri_by_area;
    final_clinical(i).main_region_area = main_region_area;
    final_clinical(i).main_region_convex_area = main_region_convex_area;
    final_clinical(i).main_region_eccentricity = main_region_eccentricity;
    final_clinical(i).main_extent = main_region_extent;
    final_clinical(i).main_region_solidity = main_region_solidity;
    final_clinical(i).main_region_perimeter = main_region_perimeter;
    final_clinical(i).main_region_angle = main_region_angle;
    final_clinical(i).main_region_peri_by_area = main_region_peri_by_area;
    final_clinical(i).main_region_major_axis = main_region_major_axis;
    final_clinical(i).main_region_minor_axis = main_region_minor_axis;


end
% 
% 

%%



% phase = 'test';
% 
% 
% load(strcat('/Users/sairam/Desktop/kid-svs/mat-files/kidney-c1/', phase, '/data.mat'));
% 
% 
% path_svs = (strcat('/Users/sairam/Desktop/kid-svs/svs-files/kidney-c1/', phase , '-c1'));
% slide_info=importdata(strcat('/Users/sairam/Desktop/kid-svs/svs-files/kidney-c1/',phase,'.txt'));
% 
% final_path = '/Users/sairam/Desktop/survival_cell_experiments/';
% 
% cd(path_svs)
% count = 0;
% counter = 0;
% 
% for sld = 1:size(files,1)
%     
%     mkdir(strcat('/Users/sairam/Desktop/survival_cell_experiments/',files(sld).name(1:12),'/',phase,'-',num2str(sld)))
%     disp(files(sld).name(1:12))
%     disp(sld)
%     if(slide_info(sld) == 0)
%         continue
%     end
%     load(strcat('/Users/sairam/Desktop/kid-svs/test-40x/mat-files-c1/', phase , '-c1-mat-files/' , num2str(sld) ,'.mat'));
%     
%     if(exist(strcat('/Users/sairam/Desktop/survival_experiments_clean/',files(sld).name(1:12),'/', phase, '-', num2str(sld),'.png')))
%     
%         image = imread(files(sld).name,1);
%         reference_image =  imread(strcat('/Users/sairam/Desktop/survival_experiments_clean/',files(sld).name(1:12),'/', phase, '-', num2str(sld),'.png'));
% 
%         for i = 1:size(scores,1)
%             if(scores(i,3) == 1)
%                 if(reference_image(scores(i,1)+100,scores(i,2)+100) > 200)
%                     saving = strcat(final_path , files(sld).name(1:12) , '/' , phase , '-' , num2str(sld), '/', phase, '-' , num2str(i) , '.png' );
%                     im = (image(scores(i,1):scores(i,1)+511,scores(i,2):scores(i,2)+511,:));
%                     im_nucleus = hmt(im);
%                     imwrite(im_nucleus, saving, 'mode', 'lossless');
%                 end
% 
%             end
%         end
%     end
%    
% 
% %imwrite(img, strcat(saving,'normal.png'), 'mode', 'lossless');
% %      if(sld ~=34)
% %        imwrite(binary_img, saving, 'mode', 'lossless');
% %        end
%     
% 
% 
% 
% end





%%


%saving morpholigically processed images



% binary_path = '/Users/sairam/Desktop/survival_experiments/';
% cd(binary_path)
% files = dir();
% files(1:3) = [];
% 
% final_path = '/Users/sairam/Desktop/survival_experiments_clean/';
% 
% 
% 
% 
% %final_clinical(537) = struct();
% 
% for i = 1:size(files,1)
%     disp(i)
%     disp(files(i).name)
% 
%     
%     
%     imgs = dir(strcat(binary_path, files(i).name ,  '/*.png'));
%     if(size(imgs,1) == 0)
%         disp('hi')
%     else
%         for j = 1:size(imgs,1)
%             img = imread(strcat(imgs(j).folder,'/',imgs(j).name));
%             CC = bwconncomp(img,8);
%             if(CC.NumObjects > 0)
%                 numPixels = cellfun(@numel,CC.PixelIdxList);
%                 removal = (numPixels > (max(numPixels)/3));
% 
%                 for idx = 1:size(removal,2)
%                     if(removal(idx) == 0)
%                         img(CC.PixelIdxList{idx}) = 0;
%                     end
%                 end
%                 
%                 imwrite(img, strcat(final_path ,imgs(j).folder(end-11:end), '/',imgs(j).name) , 'mode', 'lossless');
%                
%             end
% 
%         end          
%     end
% 
% end

%%

% binary_path = '/Users/sairam/Desktop/survival_cell_experiments/';
% cd(binary_path)
% files = dir();
% files(1:3) = [];
% 
% 
% 
% 
% final_clinical(size(files,1)) = struct();
% 
% for i = 1:size(files,1)
%     disp(i)
%     disp(files(i).name)
%     
%     total_area = 0;
%     total_convex_area = 0;
%     total_perimeter = 0;
%     total_filled_area = 0;
%     total_major_axis = 0;
%     total_minor_axis = 0;
%     total_peri_by_area = 0;
%   
%     
%     files_layer = dir(strcat(binary_path, files(i).name,'/'));
%     
%     if(strcmp(files_layer(1).name,'.') == 1)
%         files_layer(1) = [];
%     end
% 
%     if(strcmp(files_layer(1).name,'..')==1)
%         files_layer(1) = [];
%     end
%     
%     if(size(files_layer,2) > 0)
%         if(strcmp(files_layer(1).name,'.DS_Store')==1)
%             files_layer(1) = [];
%         end
%     end
% 
%     if(size(files_layer,2) > 0)
% 
%     for lay = 1:size(files_layer,1)
% 
%         disp(files_layer(lay).name)
%         imgs = dir(strcat(binary_path, files(i).name , '/' , files_layer(lay).name ,'/*.png'));
% 
%         if(size(imgs,1) == 0)
%             total_area = total_area + 0;
%             total_convex_area = total_convex_area + 0;
%             total_perimeter = total_perimeter + 0;
%             total_filled_area = total_filled_area + 0;
%             total_major_axis = total_major_axis + 0;
%             total_minor_axis = total_minor_axis + 0;
%             total_peri_by_area = total_peri_by_area + 0;
% 
%         else
%             for j = 1:size(imgs,1)
%                 img = imread(strcat(imgs(j).folder,'/',imgs(j).name));
% 
%                 CC = bwconncomp(img,8);
%                 stats = regionprops(CC,'all');
% 
%                 % total Area
%                 for ind = 1:size(stats,1)
%                     total_area = total_area + stats(ind).Area;
%                 end
%                 %Total Convex Area
%                 for ind = 1:size(stats,1)
%                     total_convex_area = total_convex_area + stats(ind).ConvexArea;
%                 end
%                 %total Perimeter
%                 for ind = 1:size(stats,1)
%                     total_perimeter = total_perimeter + stats(ind).Perimeter;
%                 end                
%                 %total Filled Area
%                 for ind = 1:size(stats,1)
%                     total_filled_area = total_filled_area + stats(ind).FilledArea;
%                 end
%                 %total major axis
%                 for ind = 1:size(stats,1)
%                     total_major_axis = total_major_axis + stats(ind).MajorAxisLength;
%                 end              
%                 %total Minor axis
%                 for ind = 1:size(stats,1)
%                     total_minor_axis = total_minor_axis + stats(ind).MinorAxisLength;
%                 end 
%                 %total peri^2 by area
%                 for ind = 1:size(stats,1)
%                     total_peri_by_area = total_peri_by_area + (stats(ind).Perimeter^2 / stats(ind).Area);
%                 end 
% 
%             end
% 
%                 
%         end
%     end
% 
%     total_area = total_area/size(files_layer,1);
%     total_convex_area = total_convex_area/size(files_layer,1);
%     total_perimeter = total_perimeter/size(files_layer,1);
%     total_filled_area = total_filled_area/size(files_layer,1);
%     total_major_axis = total_major_axis/size(files_layer,1);
%     total_minor_axis = total_minor_axis/size(files_layer,1);
%     total_peri_by_area = total_peri_by_area/size(files_layer,1);
%     
%     end
% 
%     final_clinical(i).submitter_id = files(i).name;
%     final_clinical(i).total_area = total_area;
%     final_clinical(i).total_convex_area = total_convex_area;
%     final_clinical(i).total_perimeter = total_perimeter;
%     final_clinical(i).total_filled_area = total_filled_area;
%     final_clinical(i).total_major_axis = total_major_axis;
%     final_clinical(i).total_minor_axis = total_minor_axis;
%     final_clinical(i).total_peri_by_area = total_peri_by_area;
% 
%     
% end

%writetable(struct2table(final_clinical), '/Users/sairam/Desktop/cell_features.xlsx')
%%

% phase = 'train';


% load(strcat('/Users/sairam/Desktop/kid-svs/mat-files/kidney-c3/', phase, '/data.mat'));
% path_svs = (strcat('/Volumes/Sairam/kidney-c3/', phase , '-c3'));
% final_path = '/Users/sairam/Desktop/survival_experiments-c3/';
% 
% cd(path_svs)
% 
% 
% for sld = 1:size(files,1)
%     disp(sld)
% %    img = imread(files(sld).name,1);
%     image = imread(files(sld).name,1);
%     
%     
%     saving = strcat(final_path , files(sld).name(1:12) , '/' , phase , '-' , num2str(sld) , '.png' );
% 
% %imwrite(img, strcat(saving,'normal.png'), 'mode', 'lossless');
% %      if(sld ~=34)
%         imwrite(image, saving, 'mode', 'lossless');
% %        end
%     disp(files(sld).name(1:12))
% 
% 
% 
% end


%%



% binary_path = '/Users/sairam/Desktop/survival_experiments-c3/';
% cd(binary_path)
% files = dir();
% files(1:3) = [];
% 
% cell_path = '/Users/sairam/Desktop/survival_experiments_clean-c3/';
% 
% 
% 
% final_clinical(1) = struct();
% for i = 1:size(files,1)
%     disp(i)
%     disp(files(i).name)
%     
% %     if(i~=3)
% %         continue
% %     end
%     total_area = 0;
%     total_convex_area = 0;
%     total_perimeter = 0;
%     total_filled_area = 0;
%     total_major_axis = 0;
%     total_minor_axis = 0;
%     total_peri_by_area = 0;
%     main_region_area = 0;
%     main_region_convex_area = 0;
%     main_region_eccentricity = 0;
%     main_region_extent = 0;
%     main_region_solidity = 0;
%     main_region_perimeter = 0;
%     main_region_angle = 0;
%     main_region_peri_by_area = 0;
%     main_region_major_axis = 0;
%     main_region_minor_axis = 0;
%     
%     
%     imgs = dir(strcat(binary_path, files(i).name ,  '/*.png'));
%     if(size(imgs,1) == 0)
%         total_area = 0;
%         total_convex_area = 0;
%         total_perimeter = 0;
%         total_filled_area = 0;
%         total_major_axis = 0;
%         total_minor_axis = 0;
%         total_peri_by_area = 0;
%         main_region_area = 0;
%         main_region_convex_area = 0;
%         main_region_eccentricity = 0;
%         main_region_extent = 0;
%         main_region_solidity = 0;
%         main_region_perimeter = 0;
%         main_region_angle = 0;
%         main_region_peri_by_area = 0;
%         main_region_major_axis = 0;
%         main_region_minor_axis = 0;
%         
%         
%     else
%         for j = 1:size(imgs,1)
%             
% 
%             total_area = 0;
%             total_convex_area = 0;
%             total_perimeter = 0;
%             total_filled_area = 0;
%             total_major_axis = 0;
%             total_minor_axis = 0;
%             total_peri_by_area = 0;
%             main_region_area = 0;
%             main_region_convex_area = 0;
%             main_region_eccentricity = 0;
%             main_region_extent = 0;
%             main_region_solidity = 0;
%             main_region_perimeter = 0;
%             main_region_angle = 0;
%             main_region_peri_by_area = 0;
%             main_region_major_axis = 0;
%             main_region_minor_axis = 0;
%         
%             img = imread(strcat(imgs(j).folder,'/',imgs(j).name));
%             disp(imgs(j).name)
%             CC = bwconncomp(img, 8);
%   
%                 numPixels = cellfun(@numel,CC.PixelIdxList);
%                 removal = (numPixels > (max(numPixels)/3));
% 
%                 for idx = 1:size(removal,2)
%                     if(removal(idx) == 0)
%                         img(CC.PixelIdxList{idx}) = 0;
%                     end
%                 end
%                 
%                 imwrite(img,strcat(cell_path,files(i).name,'/',imgs(j).name),'mode', 'lossless')
%                 img = imresize(img,0.25);
%                 CC = bwconncomp(img,8);
%                 
%                 clear img
%                 if(CC.NumObjects > 0)
%                     numPixels = cellfun(@numel,CC.PixelIdxList);
%                     [biggest, idx] = max(numPixels);
%                     clear numPixels
%                     stats = regionprops(CC,'all');
% 
%                     % total Area
%                     for ind = 1:size(stats,1)
%                         total_area = total_area + stats(ind).Area;
%                     end
%                     %Total Convex Area
%                     for ind = 1:size(stats,1)
%                         total_convex_area = total_convex_area + stats(ind).ConvexArea;
%                     end
%                     %total Perimeter
%                     for ind = 1:size(stats,1)
%                         total_perimeter = total_perimeter + stats(ind).Perimeter;
%                     end                
%                     %total Filled Area
%                     for ind = 1:size(stats,1)
%                         total_filled_area = total_filled_area + stats(ind).FilledArea;
%                     end
%                     %total major axis
%                     for ind = 1:size(stats,1)
%                         total_major_axis = total_major_axis + stats(ind).MajorAxisLength;
%                     end              
%                     %total Minor axis
%                     for ind = 1:size(stats,1)
%                         total_minor_axis = total_minor_axis + stats(ind).MinorAxisLength;
%                     end 
%                     %total peri^2 by area
%                     for ind = 1:size(stats,1)
%                         total_peri_by_area = total_peri_by_area + (stats(ind).Perimeter^2 / stats(ind).Area);
%                     end 
%                     %main region area
%                     main_region_area = stats(idx).Area;
%                     %main region convex area
%                     main_region_convex_area = stats(idx).ConvexArea;                
%                     %main region eccentricity
%                     main_region_eccentricity = stats(idx).Eccentricity;                
%                     %main region extent
%                     main_region_extent = stats(idx).Extent;
%                     %main region solidity
%                     main_region_solidity = stats(idx).Solidity;
%                     %main region perimeter
%                     main_region_perimeter = stats(idx).Perimeter;
%                     %main region angle
%                     main_region_angle = stats(idx).Orientation;
%                     %main region peri by area
%                     main_region_peri_by_area = ((stats(idx).Perimeter)^2 / stats(idx).Area);
%                     %main region minor axis
%                     main_region_minor_axis = stats(idx).MinorAxisLength;
%                     %major axis
%                     main_region_major_axis = stats(idx).MajorAxisLength;
%                 end
%                 
%                 final_clinical(1).submitter_id = files(i).name;
%                 final_clinical(1).total_area = total_area;
%                 final_clinical(1).total_convex_area = total_convex_area;
%                 final_clinical(1).total_perimeter = total_perimeter;
%                 final_clinical(1).total_filled_area = total_filled_area;
%                 final_clinical(1).total_major_axis = total_major_axis;
%                 final_clinical(1).total_minor_axis = total_minor_axis;
%                 final_clinical(1).total_peri_by_area = total_peri_by_area;
%                 final_clinical(1).main_region_area = main_region_area;
%                 final_clinical(1).main_region_convex_area = main_region_convex_area;
%                 final_clinical(1).main_region_eccentricity = main_region_eccentricity;
%                 final_clinical(1).main_extent = main_region_extent;
%                 final_clinical(1).main_region_solidity = main_region_solidity;
%                 final_clinical(1).main_region_perimeter = main_region_perimeter;
%                 final_clinical(1).main_region_angle = main_region_angle;
%                 final_clinical(1).main_region_peri_by_area = main_region_peri_by_area;
%                 final_clinical(1).main_region_major_axis = main_region_major_axis;
% 
%                 
%                  
%                 writetable(struct2table(final_clinical),strcat(imgs(j).folder,'/resized-',num2str(j),'.xlsx'))
%                 clear final_clinical
%                 final_clinical(1) = struct();
                
%             end
% 
% 
%     end            
% end


%%


% binary_path = '/Users/sairam/Desktop/survival_experiments-c3/';
% cd(binary_path)
% files = dir();
% files(1:3) = [];
% 
% 
% 
% 
% 
% final_clinical(size(files,1)) = struct();
% for i = 1:size(files,1)
%     count = 0;
%     disp(i)
%     disp(files(i).name)
%     
%     total_area = 0;
%     total_convex_area = 0;
%     total_perimeter = 0;
%     total_filled_area = 0;
%     total_major_axis = 0;
%     total_minor_axis = 0;
%     total_peri_by_area = 0;
%     main_region_area = 0;
%     main_region_convex_area = 0;
%     main_region_eccentricity = 0;
%     main_region_extent = 0;
%     main_region_solidity = 0;
%     main_region_perimeter = 0;
%     main_region_angle = 0;
%     main_region_peri_by_area = 0;
%     main_region_major_axis = 0;
%     main_region_minor_axis = 0;
%     
% 
%     imgs = dir(strcat(binary_path, files(i).name ,  '/*.xlsx'));
%     if(size(imgs,1) == 0)
%         total_area = 0;
%         total_convex_area = 0;
%         total_perimeter = 0;
%         total_filled_area = 0;
%         total_major_axis = 0;
%         total_minor_axis = 0;
%         total_peri_by_area = 0;
%         main_region_area = 0;
%         main_region_convex_area = 0;
%         main_region_eccentricity = 0;
%         main_region_extent = 0;
%         main_region_solidity = 0;
%         main_region_perimeter = 0;
%         main_region_angle = 0;
%         main_region_peri_by_area = 0;
%         main_region_major_axis = 0;
%         main_region_minor_axis = 0;
%         
%         
%     else
%         for j = 1:size(imgs,1)
%             %a(2).name(1)
%             
%            
%             
%             if(strcmp(imgs(j).name(1),'r')==0)
%                 continue
%             else
%                 img = xlsread(strcat(imgs(j).folder,'/',imgs(j).name));
%                 count = count + 1;
%             end
%             
% 
%             
%             total_area = total_area + img(1);
%             total_convex_area = total_convex_area + img(2);
%             total_perimeter = total_perimeter + img(3);
%             total_filled_area = total_filled_area + img(4);
%             total_major_axis = total_major_axis + img(5);
%             total_minor_axis = total_minor_axis + img(6);
%             total_peri_by_area = total_peri_by_area + img(7);
%             main_region_area = main_region_area + img(8);
%             main_region_convex_area = main_region_convex_area + img(9);
%             main_region_eccentricity = main_region_eccentricity + img(10);
%             main_region_extent = main_region_extent + img(11);
%             main_region_solidity = main_region_solidity + img(12);
%             main_region_perimeter = main_region_perimeter + img(13);
%             main_region_angle = main_region_angle + img(14);
%             main_region_peri_by_area =  main_region_peri_by_area + img(15);
%             main_region_major_axis = main_region_major_axis + img(16);
%             main_region_minor_axis = 0;
% 
% 
%                 
%         end
%         
%         total_area = total_area/count;
%         total_convex_area = total_convex_area/count;
%         total_perimeter = total_perimeter/count;
%         total_filled_area = total_filled_area/count;
%         total_major_axis = total_major_axis/count;
%         total_minor_axis = total_minor_axis/count;
%         total_peri_by_area = total_peri_by_area/count;
%         main_region_area = main_region_area/count;
%         main_region_convex_area = main_region_convex_area/count;
%         main_region_eccentricity = main_region_eccentricity/count;
%         main_region_extent = main_region_extent/count;
%         main_region_solidity = main_region_solidity/count;
%         main_region_perimeter = main_region_perimeter/count;
%         main_region_angle = main_region_angle/count;
%         main_region_peri_by_area = main_region_peri_by_area/count;
%         main_region_major_axis = main_region_major_axis/count;
%         main_region_minor_axis = main_region_minor_axis/count;
%     
% 
% 
% 
%     end
%     
%     
%     final_clinical(i).submitter_id = files(i).name;
%     final_clinical(i).total_area = (total_area)*16;
%     final_clinical(i).total_convex_area = (total_convex_area)*16;
%     final_clinical(i).total_perimeter = (total_perimeter)*4;
%     final_clinical(i).total_filled_area = (total_filled_area)*16;
%     final_clinical(i).total_major_axis = (total_major_axis)*4;
%     final_clinical(i).total_minor_axis = (total_minor_axis)*4;
%     final_clinical(i).total_peri_by_area = total_peri_by_area;
%     final_clinical(i).main_region_area = main_region_area*16;
%     final_clinical(i).main_region_convex_area = main_region_convex_area*16;
%     final_clinical(i).main_region_eccentricity = main_region_eccentricity;
%     final_clinical(i).main_extent = main_region_extent;
%     final_clinical(i).main_region_solidity = main_region_solidity;
%     final_clinical(i).main_region_perimeter = main_region_perimeter*4;
%     final_clinical(i).main_region_angle = main_region_angle;
%     final_clinical(i).main_region_peri_by_area = main_region_peri_by_area;
%     final_clinical(i).main_region_major_axis = main_region_major_axis*4;
%     final_clinical(i).main_region_minor_axis = main_region_minor_axis*4;
%     
% end
% 
% writetable(struct2table(final_clinical),strcat('/Users/sairam/Desktop/final_clinical-c3.xlsx'))


%%



% phase = 'train';
% 
% 
% load(strcat('/Users/sairam/Desktop/kid-svs/mat-files/kidney-c1/', phase, '/data.mat'));
% 
% 
% path_svs = (strcat('/Volumes/Sairam/kidney-c1/', phase , '-c1'));
% 
% classes = importdata(strcat('/Users/sairam/Desktop/kidney/cancer_normal_40x/class1/resnet-18/',phase,'/classes.txt'));
% probability = importdata(strcat('/Users/sairam/Desktop/kidney/cancer_normal_40x/class1/resnet-18/',phase,'/probabilities.txt'));
% 
% slide_info=importdata(strcat('/Users/sairam/Desktop/kid-svs/svs-files/kidney-c1/',phase,'.txt'));
% 
% final_path = '/Users/sairam/Desktop/survival_experiments_patches/';
% prob = 1.0 ./ ( 1.0 + exp(-probability));
% 
% if(strcmp(phase,'test')==1)
%     index = 126294;
% elseif(strcmp(phase,'valid') == 1)
%     index = 78663;
% elseif(strcmp(phase,'train') == 1)
%     index = 374812;
% end
% 
% 
% 
% 
% 
% cd(path_svs)
% count = 0;
% counter = 0;
% svs_files = dir('**/*.svs');
% 
% for sld = 1:size(files,1)
%     disp(sld)
%     disp(files(sld).name(1:12))
%     if(slide_info(sld) == 0)
%         continue
%     end
%     load(strcat('/Users/sairam/Desktop/kid-svs/test-40x/mat-files-c1/', phase , '-c1-mat-files/' , num2str(sld) ,'.mat'));
%     
%     
% %    img = imread(files(sld).name,1);
%     for gi = 1:size(svs_files,1)
%         if(strcmp(files(sld).name , svs_files(gi).name) == 1)
%             img = imread(svs_files(gi).name,1);
% 
%         end
%     end
% 
% 
% 
%     for i = 1:size(scores,1)
%         if(scores(i,3) == 1)
%             count = count + 1;
%             if(prob(index+count) > 0.8)
%                 if(classes(index+count)  == 1)
% %                    binary_img(scores(i,1): scores(i,1) + 511, scores(i,2): scores(i,2) + 511) = binary_img(scores(i,1): scores(i,1) + 511, scores(i,2): scores(i,2) + 511) + 255;
%                     im = img(scores(i,1): scores(i,1) + 511, scores(i,2): scores(i,2) + 511);
%                     saving = strcat(final_path , files(sld).name(1:12) , '/' , phase , '-' , num2str(i) , '.png');
%                     imwrite(binary_img, saving, 'mode', 'lossless');
%                     counter = counter + 1;
%                 end
%  %               disp(counter)
%             end
%         
%         
%         end
%     
%     end
% 
% 
% %imwrite(img, strcat(saving,'normal.png'), 'mode', 'lossless');
% %      if(sld ~=34)
% %         imwrite(binary_img, saving, 'mode', 'lossless');
% %        end
%     disp(files(sld).name(1:12))
% 
% 
% 
% end