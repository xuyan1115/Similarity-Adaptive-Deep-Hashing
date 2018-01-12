% ------------------------------------------------------------------------
function images = prepare_batch(image_files, root_folder, mean_data, batch_size)
% ------------------------------------------------------------------------
if nargin < 3
    % load mean file
    d = load('ilsvrc_2012_mean.mat');
    mean_data = d.mean_data;
end

num_images = length(image_files);
if nargin < 4
    batch_size = num_images;
end

IMAGE_DIM = 256;
CROPPED_DIM = 224;

indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
center = floor(indices(2) / 2)+1;
num_images = length(image_files);
images = zeros(CROPPED_DIM,CROPPED_DIM,3,batch_size,'single');

parfor i=1:num_images
    % read file
    img_file = fullfile(root_folder, image_files{i})
    fprintf('%c Preparing %s\n',13, img_file);
    try
    im = imread(img_file);
    % resize to fixed input size
    im = single(im);
    im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    % Transform GRAY to RGB
    if size(im,3) == 1
        im = cat(3,im,im,im);
    end
    % permute from RGB to BGR (IMAGE_MEAN is already BGR)
    im = im(:,:,[3 2 1]) - mean_data;
   % im = im(:,:,[3 2 1]);
    % Crop the center of the image
    images(:,:,:,i) = permute(im(center:center+CROPPED_DIM-1,...
    center:center+CROPPED_DIM-1,:),[2 1 3]);
    catch
        warning('Problems with file',image_files{i});
    end
end
