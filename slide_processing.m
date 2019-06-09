%Removing all the svs files from their folders

% classes = ['KIRC'];
% 
% 
% 
% 
% for class = 1:size(classes,1)
%     
% 
% 
%     disp(classes(class,:))
%     saved_svs_path = strcat('/shared/supernova/home/sairam.tabibu/kidney-slides/',classes(class,:));
%     cd(saved_svs_path);
%     final_svs_path = strcat('/shared/supernova/home/sairam.tabibu/kidney-slides/',classes(class,:),'-slides/');
%     files = dir;
%     directoryNames = {files([files.isdir]).name};
%     directoryNames = directoryNames(~ismember(directoryNames,{'.','..'}));
% 
%     total = length(directoryNames);
%     
%     for slides = 1:total
%         disp(slides)
%         cd(char(directoryNames(slides)));
%         current_svs = dir('*.svs');
%         movefile(current_svs(1).name,final_svs_path);
%         cd ..
%     end
% 
% end



%%
% Code to remove unreadable files and distinguish between Normal and
% Cancerous slides


% a=dir('*.svs');
% for i=1:size(a,1)
%     disp(i)
%     img=imread(strcat(a(i).folder,'/',a(i).name),3);
%     if (sum(sum(sum(img(100:200,100:200,:)))) == 0)
%         movefile(strcat(a(i).folder,'/',a(i).name),'unreadable')
%     else
%         if(str2num(a(i).name(14:15)) > 10)
%             movefile(strcat(a(i).folder,'/',a(i).name),'normal')
%         else
%             movefile(strcat(a(i).folder,'/',a(i).name),'cancer')
%         end
%     end


%%

% Sorting them in train, validation and test folders


classes = ['KIRP','KIRC','KICH'];

mat_files_path = '/shared/supernova/home/sairam.tabibu/mat-files/';
fold = ['train';'valid';'testi'];

for class = 1:size(classes,1)
    disp(classes(class,:))
    final_svs_path = strcat('/shared/supernova/home/sairam.tabibu/kidney-slides/',classes(class,:),'-slides/');
    cd(final_svs_path)
    svs_files = dir('*.svs');
    
    if(strcmp(classes(class,:),'KIRC')==1)
        mode_index = 1;
    elseif(strcmp(classes(class,:),'KIRP')==1)
        mode_index = 2;
    else
        mode_index = 3;
    end
    
    for mode = 1:size(fold,1)
        disp(fold(mode,:))
        load(strcat(mat_files_path,'kidney-c',num2str(mode_index),'/',fold(mode,:),'/data.mat'));
        
        for j = 1:size(files,1) 
            disp(j)
            for k=1:size(svs_files,1)
                if(strcmp(svs_files(k).name , files(j).name) == 1)
                    final = strcat(final_svs_path,fold(mode,:),'-c',num2str(mode_index),'/');
                    movefile(svs_files(k).name,final);
                    
                end
            end
        end
        
        
        
    end
    
end


%%



