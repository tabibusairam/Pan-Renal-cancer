    
path='/Users/sairam/Desktop/kid-svs/kidney-c';
path1 = '/Users/sairam/Desktop/kid-svs/mat-files/kidney-c';
path2 = '/Users/sairam/Desktop/kid-svs/svs-files/kidney-c';



for class =1:3
    mkdir(strcat(path,num2str(class),'/mat_files'))
    load(strcat(path1,num2str(class),'/test/','data.mat'));
    
    current_path = strcat(path2,num2str(class));
    disp(current_path)
    cd(current_path)
    
    svs_files = dir('**/*.svs');
    
    
    
    for fi=1:size(files,1)
        for gi = 1:size(svs_files,1)
            if((strcmp(files(fi).name,svs_files(gi).name) == 1))
                disp(fi)
                image = imread(strcat(svs_files(gi).folder,'/',svs_files(gi).name),3);
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
                %tag = strcat(path2, num2str(fi),'-',num2str(k),'.png');
                %imwrite(im, tag, 'mode', 'lossless');
            end
        end
        tag1 = strcat(path,num2str(class),'/mat_files/',num2str(fi),'.mat');
        save(tag1,'scores');

    end



end





