% This file uses to load picture data from txt file
clear all;
% train data
train_struct = importdata('./cifar10_train_flip.txt');
train_path = train_struct.textdata;
traingnd = single(train_struct.data);
clear train_struct;

% test data
test_struct = importdata('test/test_list1.txt');
test_path = test_struct.textdata;
testgnd = single(test_struct.data);
clear test_struct;

traindata = zeros(3072, length(train_path));
testdata = zeros(3072, length(test_path));

for i = 1:length(train_path)
   if mod(i, 2000) == 0
       fprintf('%d\n', i);
   end
   temp = imread(train_path{i});
   traindata(:,i) = temp(:);
end

for i = 1:length(test_path)
   if mod(i, 1000) == 0
       fprintf('%d\n', i);
   end
   temp = imread(strcat('test/',test_path{i}));
   testdata(:,i) = temp(:);
end

save('cifar10', 'traindata', 'traingnd', 'testdata', 'testgnd', '-v7.3');
