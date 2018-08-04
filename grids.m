
files = dir('**/*.svs');
image = imread(strcat(files(1).folder,'/',files(1).name),3);
imshow(image)
hold on
[row, col, cha] = size(image);
i=1; j=1;k=1;
while(i<row-511)
    j=1;
    while(j<col-511)
        scores(k,:) = [i,j,0];
        k = k+1;
        j=j+255;
        rectangle('Position',[j j+511 i i+511],'EdgeColor','b');
    end
    i=i+255;
end