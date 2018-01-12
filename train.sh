#!/bin/bash

#set -x
set -e

if [ $# -ne 3 ] ; then
    echo 'Please input parameters CAFFE_ROOT ROOT_FOLDER and GPU_ID'
    exit 1;
fi

CAFFE_ROOT=$1
ROOT_FOLDER=$2  # image root folder
GPU_ID=$3

# iteration 1
echo "extract fc7 features"
cd fc7_features
python extract_features.py vgg_16 ../caffemodels/VGG_ILSVRC_16_layers.caffemodel ../$ROOT_FOLDER ../cifar10/cifar10_train_flip_shuffle.txt fc7 traindata
cd ..
echo "generate .mat file -> ./fc7_features/traindata.txt"

echo "update anchors by ADMM algorithm"
matlab -nojvm -nodesktop -r "run ./ADMM/ADMM.m; quit;"
echo "generate .h5 file -> ./data_from_ADMM/B_16bits.h5"

echo "finetune VGG model to initialize W."
cd finetune_network
$CAFFE_ROOT/build/tools/caffe train -solver ./solver_16bits.prototxt -weights ../caffemodels/VGG_ILSVRC_16_layers.caffemodel -gpu $GPU_ID
cd ..
echo "finetuning finished!"


# iteration 2
echo "extract fc7 features"
cd fc7_features
python extract_features.py vgg_16 ../caffemodels/iter/SADH_16bits_iter_70800.caffemodel ../$ROOT_FOLDER ../cifar10/cifar10_train_flip_shuffle.txt fc7 traindata1
cd ..
echo "generate .mat file -> ./fc7_features/traindata1.txt"

echo "update anchors by ADMM algorithm"
matlab -nojvm -nodesktop -r "run ./ADMM/ADMM_1.m; quit;"
echo "generate .h5 file -> ./data_from_ADMM/B_16bits_1.h5"

echo "finetune VGG model to initialize W."
cd finetune_network
$CAFFE_ROOT/build/tools/caffe train -solver ./solver_16bits_1.prototxt -weights ../caffemodels/iter/SADH_16bits_iter_70800.caffemodel -gpu $GPU_ID
cd ..
echo "fintuning finished!"


# iteration 3
echo "extract fc7 features"
cd fc7_features
python extract_features.py vgg_16 ../caffemodels/iter1/SADH_16bits_iter_70800.caffemodel ../$ROOT_FOLDER ../cifar10/cifar10_train_flip_shuffle.txt fc7 traindata2
cd ..
echo "generate .mat file -> ./fc7_features/traindata2.txt"

echo "update anchors by ADMM algorithm"
matlab -nojvm -nodesktop -r "run ./ADMM/ADMM_2.m; quit;"
echo "generate .h5 file -> ./data_from_ADMM/B_16bits_2.h5"

echo "finetune VGG model to initialize W."
cd finetune_network
$CAFFE_ROOT/build/tools/caffe train -solver ./solver_16bits_2.prototxt -weights ../caffemodels/iter1/SADH_16bits_iter_70800.caffemodel -gpu $GPU_ID
cd ..
echo "fintuning finished!"

echo "test"
matlab -nojvm -nodesktop -r "run ./run_cifar10_16bits.m; quit;"
echo "finished!"
