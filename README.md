Similarity-Adaptive Deep Hashing (SADH)
====

Unsupervised Deep Hashing with Similarity-Adaptive and Discrete Optimization

Created by Fumin Shen, Yan Xu, Li Liu, Yang Yang, Zi Huang, Heng Tao Shen

The details can be found in the [TPAMI 2018 paper](http://cfm.uestc.edu.cn/~fshen/SADH.pdf).

## Contents ##

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Resources](#resources)
- [Citation](#citation)

### Prerequisites ###

1. Requirements for `Caffe`, `pycaffe` and `matcaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html)).

2. Prerequisites for datasets.
    
    **Note:** In our experiments, we horizontally flip training images manually for data augmentation. If the size of your training data is small (< 100K, like CIFAR-10. MNIST), you should do this step.

    We also provide our flipping code in `cifar10/flip_img.m`, you can run it to handle your own datasets.

3. VGG-16 pre-trained model on ILSVC12 datasets, and save it in `caffemodels` directory.
    

### Installation ###

Enter caffe directory and download the source codes.
```Shell
    cd caffe/
```

Modify `Makefile.config` and build Caffe with following commands:
```Shell
    make all -j8
    make pycaffe
    make matcaffe
```

### Usage ###

We only supply the code to train 16-bit SADH on CIFAR-10 dataset.

We integrate train step and test step in a bash file `train.sh`, please run it as follows:
```Shell
    ./train.sh [ROOT_FOLDER] [GPU_ID]
    # ROOT_FOLDER is the root folder of image datasets, e.g. ./cifar10/
    # GPU_ID is the GPU you want to train on
```

### Resources ###

We supply CIFAR-10 dataset, which has been flipped. You can download it by following links:

- CIFAR-10 dataset (png format): [BaiduYun](https://pan.baidu.com/s/1azKt0WT4pIyPuiximHiTZw) (Updated).


### Citation ###

If you find our approach useful in your research, please consider citing:

    @article{'shen2018tpami',
        author   = {Fumin Shen and Yan Xu and Li Liu and Yang Yang and Zi Huang and Heng Tao Shen},
        journal  = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)}, 
        title    = {Unsupervised Deep Hashing with Similarity-Adaptive and Discrete Optimization},
        year     = {2018}
    }
