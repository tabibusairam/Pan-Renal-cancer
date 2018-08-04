    
mode = ['rectum-test'];
%classes = ['above8'];

path = '/Users/user/Desktop/colorectal-svs/rectum/rectum-test';
path2 = '/Users/user/Desktop/colorectal-1/rectum/rectum-test/';



disp(mode(1,:))
%disp(classes(1,:))
%path1 = strcat(path,'/',mode(1,:));
cd(path);
files = dir('**/*.svs');
save('data.mat','files');



for fi=1:size(files,1)

    disp(fi)
    image = imread(strcat(files(fi).folder,'/',files(fi).name),3);


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

        if((sum(sum(im_R)) > 200000) && (sum(sum(im_G)) > 200000) && (sum(sum(im_B)) > 200000))
            scores(k,3) = 0;
        else
            scores(k,3) = 1;
            tag = strcat(path2, num2str(fi),'-',num2str(k),'.png');
            imwrite(im, tag, 'mode', 'lossless');
        end
    end
    tag1 = strcat(num2str(fi),'.mat');
    save(tag1,'scores');

end









