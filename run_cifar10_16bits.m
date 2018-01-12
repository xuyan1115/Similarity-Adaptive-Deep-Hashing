close all;
clear;

addpath(genpath('caffe/'));
addpath(genpath('utils/'))
% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu = 1;

% top K returned images
top_k = 1000;
root_folder = './cifar10';
feat_len = 16;
dataset = 'cifar';
topk = 1:2000;

% set result folder
result_folder = './analysis';

% models
model_file = './caffemodels/iter2/SADH_16bits_iter_59000.caffemodel';
% model definition
model_def_file = './caffemodels/vgg_feature16.prototxt';

% train-test
test_file_list = './cifar10/test-file-list.txt';
test_label_file = './cifar10/test-label.txt';
train_file_list = './cifar10/train-file-list.txt';
train_label_file = './cifar10/train-label.txt';
% --- settings end here ---

% caffe mode setting
phase = 'test'; % run with phase test (so that dropout isn't applied)


% --- settings end here ---

% outputs
feat_test_file = sprintf('%s/feat-test.mat', result_folder);
feat_train_file = sprintf('%s/feat-train.mat', result_folder);
binary_test_file = sprintf('%s/binary-test.mat', result_folder);
binary_train_file = sprintf('%s/binary-train.mat', result_folder);

% map and precision outputs
mean_th = 0;

% feature extraction- training set
if exist(feat_train_file, 'file') ~= 0
    load(feat_train_file);
    mean_bin = mean(feat_train');
    mean_th = mean(mean_bin);
    binary_train = (feat_train>mean_th);
else
    feat_train = feat_batch(use_gpu, model_def_file, model_file, train_file_list, root_folder, feat_len);
    save(feat_train_file, 'feat_train', '-v7.3');
    mean_bin = mean(feat_train');
    mean_th = mean(mean_bin);
    binary_train = (feat_train>mean_th);
    save(binary_train_file,'binary_train','-v7.3');
end


% feature extraction- test set
if exist(feat_test_file, 'file') ~= 0
    load(feat_test_file);
    binary_test = (feat_test>mean_th);    
else
    feat_test = feat_batch(use_gpu, model_def_file, model_file, test_file_list, root_folder, feat_len);
    save(feat_test_file, 'feat_test', '-v7.3');
    binary_test = (feat_test>mean_th);
    save(binary_test_file,'binary_test','-v7.3');
end

trn_label = load(train_label_file);
tst_label = load(test_label_file);

hammRadius = 2;
traingnd = trn_label;
testgnd = tst_label;
B = compactbit(binary_train');
tB = compactbit(binary_test');
cateTrainTest = bsxfun(@eq, traingnd, testgnd');

hammTrainTest = hammingDist(tB, B)';
% hash lookup: precision and reall
Ret = (hammTrainTest <= hammRadius+0.00001);
[Pre, Rec] = evaluate_macro(cateTrainTest, Ret)

% hamming ranking: MAP
[~, HammingRank]=sort(hammTrainTest, 1);
MAP = cat_apcal(traingnd, testgnd, HammingRank)
[Pre_500, ~] = cat_ap_topK(cateTrainTest, HammingRank, 500)
[Pre_2000, ~] = cat_ap_topK(cateTrainTest, HammingRank, 2000)
[Pre_5000, ~] = cat_ap_topK(cateTrainTest, HammingRank, 5000)


%[pre, rec] = evaluate_HammingRanking_similarity(cateTrainTest, HammingRank);
%save([result_folder, '/Deep_', dataset, '_PR_', num2str(feat_len)], 'pre', 'rec');

%[precision_at_k, ~] = cat_ap_topK(cateTrainTest, HammingRank, topk);
%save(['Deep_', dataset, '_precision_at_k_', num2str(feat_len)], 'precision_at_k');
