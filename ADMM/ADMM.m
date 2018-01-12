clear all;

addpath(genpath('minFunc_2012/'));
addpath(genpath('../utils/'));
dataset = 'cifar10';
bit = 16;
use_kmeans = 1;
folder_name = '../data_from_ADMM';
if ~exist(folder_name, 'dir')
    mkdir(folder_name)
end
save_path = sprintf('%s/B_%dbits.h5', folder_name, bit);
fprintf('---------------------------------------\n');
fprintf('save path is %s\n', save_path);
fprintf('---------------------------------------\n');
fprintf('load dataset %s...\n', dataset);
load('../fc7_features/traindata.txt');
%load './cifar_10_gist.mat';
traindata = double(traindata');
%testdata = double(testdata');
fprintf('Finished!\n');                                                         
fprintf('---------------------------------------\n'); 
fprintf('data prepocessing...\n');
X = traindata; % type: single
X = normalize(X');
clear traingnd testgnd;
traindata = traindata';
sampleMean = mean(traindata, 1);                                                       
traindata = (traindata - repmat(sampleMean, size(traindata, 1), 1));
%clear traindata testdata;
% Xlabel = traingnd;
% Ylabel = testgnd;
[n, dim] = size(X);
fprintf('Finished!\n');
fprintf('---------------------------------------\n');

% parameters
rho1 = 1e-2;
rho2 = 1e-2;
rho3 = 1e-3;
rho4 = 1e-3;
gamma = 1e-3;
sigma = 0;
max_iter = 50;
n_anchors = 500;
s = 2;     % number of nearest anchors, please tune this parameter on different datasets
if ~use_kmeans
     anchor = traindata(randsample(n, n_anchors),:);
else
    fprintf('K-means clustering to get m anchor points\n');
    [~, anchor] = litekmeans(traindata, n_anchors, 'MaxIter', 30);
    fprintf('anchor points have been selected!\n');
    fprintf('---------------------------------------\n');
end

options = [];
options.Display = 'off';
options.MaxFunEvals = 20;
options.Method = 'lbfgs';   % pcg lbfgs

% define ground-truth neighbors
fprintf('Generating anchor graphs\n');
Z = zeros(n, n_anchors);
Dis = sqdist(traindata', anchor');
%clear X;    
clear traindata;
clear anchor;

val = zeros(n, s);
pos = val;
for i = 1:s
    [val(:,i), pos(:,i)] = min(Dis, [], 2);
    tep = (pos(:,i) - 1) * n + [1:n]';
    Dis(tep) = 1e60;
end
clear Dis;
clear tep;

if sigma == 0
    sigma = mean(val(:,s) .^ 0.5);
end
val = exp(-val / (1 / 1 * sigma ^ 2));
val = repmat(sum(val, 2).^ -1, 1, s) .* val;
tep = (pos - 1) * n + repmat([1:n]', 1, s);
Z([tep]) = [val];
clear tep;
clear val;
clear pos;
% Z = sparse(Z);
lamda = sum(Z);
lambda = diag(lamda .^ -1);
size(lambda)
size(Z)
clear lamda
fprintf('Finished!\n');
fprintf('---------------------------------------\n'); 

%initization
%load('../init_32bits_B/final_32bits.mat');
%B = double(feat_train');
%B = sign(B); 
B = sign(randn(n, bit));  % -1 or 1 random number
init_B = B;
Z1 = B; Z2 = B;
Y1 = rand(n, bit);
Y2 = rand(n, bit);
one_vector = ones(n, 1);
theta1 = zeros(n, bit);
theta2 = zeros(n, bit);
loss_old = 0;
i = 0;

for i = 1:max_iter
%while true
    %i = i + 1;
    fprintf('iteration %3d\n', i);
    % update B
    tic;
    constant = Y1 + Y2 - rho1 * Z1 - rho2 * Z2;
    [Bk_tep, ~, ~, ~] = minFunc(@gradientB, B(:), options, constant, Z, lambda, rho1, rho2, rho3, rho4);
    time = toc;
    count = sum(Bk_tep > 0);
    fprintf('+1, -1: %.2f%%\n', count/n/bit*100);
    fprintf('Update B cost %f seconds\n', time); 
    Bk = reshape(Bk_tep, [n, bit]);
    fprintf('res(init_B and Bk): %d\n', sum(sign(Bk(:))-init_B(:)))

    % update Z1
    tic;
    theta1 = Bk + 1/rho1 * Y1;
    theta1(theta1 > 1) = 1;
    theta1(theta1 < -1) = -1;
%    fprintf('norm theta1(Z1) is %d\n', norm(theta1, 'fro'));
    Z1_k = theta1;

    % update Z2
    theta2 = Bk + 1/rho2 * Y2;
    norm_B = norm(theta2, 'fro');
%    fprintf('B''s norm is %d\n', norm(B, 'fro'));
%    fprintf('norm theta2 is %d\n', norm(theta2, 'fro'));
    theta2 = sqrt(n*bit) * theta2 / norm_B; 
    Z2_k = theta2;
%    fprintf('norm Z2 is %d\n', norm(Z2_k, 'fro'));
    time = toc;
    fprintf('Update Z1 and Z2 cost %f seconds\n', time);  

    %update Y1 and Y2
    tic;
    Y1_k = Y1 + gamma * rho1 * (Bk - Z1_k);
    Y2_k = Y2 + gamma * rho2 * (Bk - Z2_k);
%    fprintf('norm Y2_k is %d\n', norm(Y2_k, 'fro'));
    time = toc;
    fprintf('Update Y1 and Y2 cost %f seconds\n', time);

    B = Bk; 
    Z1 = Z1_k; Z2 = Z2_k; 
    Y1 = Y1_k; Y2 = Y2_k;
      
    res1 = B - Z1; res2 = B - Z2; tmp1 = one_vector'*B; tmp2 = B'*B-n*eye(bit,bit);
    loss = trace(B'*B-B'*Z*lambda*(Z'*B)+Y1'*res1+Y2'*res2)+rho1/2*trace(res1'*res1)...
           +rho2/2*trace(res2'*res2)+rho3/2*trace(tmp1*tmp1')+rho4/4*trace(tmp2'*tmp2);
    res = (loss - loss_old)/loss_old;
    loss_old = loss;
    fprintf('loss is %.4f, residual error is %.5f\n', loss, res);
    fprintf('---------------------------------------\n'); 
    if (abs(res) <= 1e-4)
        break;
    end
end

%clear Y1 Y1_k Y2 Y2_k Z1 Z1_k Z2 Z2_k;

final_B = B;
final_B = sign(final_B);

fprintf('save B and final_B as HDF5 file\n');
fprintf('save path is %s\n' ,save_path);
h5create(save_path, '/final_B',[size(final_B, 2) size(final_B, 1)]);
h5create(save_path, '/B',[size(B, 2) size(B, 1)]);
h5write(save_path, '/final_B', final_B');
h5write(save_path, '/B', B');
fprintf('Finished!\n');                                                         
fprintf('---------------------------------------\n'); 
