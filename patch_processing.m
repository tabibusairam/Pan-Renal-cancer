class = 'KIRP';
mode = 'valid';

if(strcmp(class,'KIRC')==1)
    mode_index = 1;
elseif(strcmp(class,'KIRP')==1)
    mode_index = 2;
else
    mode_index = 3;
end

disp(class)
disp(mode)

load(strcat('/shared/supernova/home/sairam.tabibu/mat-files/kidney-c',num2str(mode_index),'/',mode,'/data.mat'));

path = strcat('/shared/supernova/home/sairam.tabibu/kidney-slides/',class,'-slides/',mode,'-c',num2str(mode_index),'/');
cd(path)
svs_files = dir('*.svs');
final_path = strcat('/shared/supernova/home/sairam.tabibu/kidney-patches/',class,'/',mode,'-c',num2str(mode_index),'/');
mat_path = strcat('/shared/supernova/home/sairam.tabibu/kidney-patches/',class,'-mat/',mode,'-c',num2str(mode_index),'/');

count = 0;
pre = '000';
suf = '000000';


for fi=1:size(files,1)
    disp(fi)
%     
%     if(fi<4)
%         continue
%     end
    for gi = 1:size(svs_files,1)
        if(strcmp(files(fi).name , svs_files(gi).name) == 1)
            image = imread(svs_files(gi).name,3);
        end
    end
   
    [row, col, cha] = size(image);
    i=1; j=1;k=1;
    scores=[];
    while(i<row-511)
        j=1;
        while(j<col-511)
            scores(k,:) = [i,j,0];
            k = k+1;
            j=j+255;
        end
        i=i+255;
    end

    for k=1:size(scores,1)
        im = (image(scores(k,1):scores(k,1)+511,scores(k,2):scores(k,2)+511,:));
        im_R = (im(:,:,1) > 210);
        im_G = (im(:,:,2) > 210);
        im_B = (im(:,:,3) > 210);

        if((sum(sum(im_R)) > 160000) && (sum(sum(im_G)) > 160000) && (sum(sum(im_B)) > 160000))
            scores(k,3) = 0;
        else
            scores(k,3) = 1;
            
            pre_num = 3 - length(num2str(fi));
            suf_num = 6 - length(num2str(k));
            tag = strcat(final_path , pre(1:pre_num) , num2str(fi),'-',suf(1:suf_num), num2str(k),'.png');
            imwrite(im, tag, 'mode', 'lossless');
            count = count + 1;
        end
    end
    tag1 = strcat(mat_path, num2str(fi),'.mat');
    save(tag1,'scores');
    disp(count)
end