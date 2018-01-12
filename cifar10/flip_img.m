clear all;

dataset = 'cifar10';
txtfile1 = sprintf('./%s_train_flip.txt', dataset);
fid1 = fopen(txtfile1, 'wt');

txtfile2 = sprintf('./train/train_list.txt');
fid2 = fopen(txtfile2, 'r');

train_temp = importdata(txtfile2);
list_im = train_temp.textdata;
train_label = train_temp.data;
clear train_temp;

path_label = read_cell('./train/train_list1.txt');

dataDir = './train_flip/';
if exist(dataDir, 'dir') == 0
    mkdir(dataDir);
end

[data_num, ~] = size(list_im);

for i = 1:data_num
    if mod(i, 1000) == 0
        fprintf('Totally %d images have been fliped.\n', i);
    end
    img_path = list_im{i};
    img = imread(img_path);
    img_dirs = strsplit(img_path, '/');
    classDir = sprintf('./train_flip/%s', img_dirs{2});
    if exist(classDir, 'dir') == 0
        mkdir(classDir);
    end
    f = sprintf('%s/%s', classDir, img_dirs{3});
    img2 = flip(img, 2);  % flip horizontal
    imwrite(img2, f);

    %% write original image and flipped image to txt file. 
    original_img_path = strcat('train/', path_label{i});
    flip_img_path = strcat('train_flip/', path_label{i});
    fprintf(fid1, '%s\n%s\n', original_img_path, flip_img_path);
end



fclose(fid1);
fclose(fid2);
