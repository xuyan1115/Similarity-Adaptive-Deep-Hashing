#! /usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'XuYan'

import numpy as np
import caffe
import time
import sys
import copy
import os
import matplotlib.pyplot as plt
import scipy
import scipy.io

class CNN():
    def __init__(self, cnn_type, caffemodel):
        current_path = os.path.split(os.path.realpath(__file__))[0]
        if cnn_type == 'vgg_16':
            deploy = '../caffemodels/vgg_feature16.prototxt'
            mean = np.array([103.939, 116.779, 123.68])
            self.__config(deploy, caffemodel, mean, 224, 224)
        else:
            raise BaseException('''we only provide vgg_16''', cnn_type)

    def __config(self, deploy, caffemodel, mean, width, height):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.net = caffe.Net(deploy, caffemodel, caffe.TEST)

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', mean)
        self.transformer.set_raw_scale('data', 255)
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.net.blobs['data'].reshape(1, 3, width, height)

    def __saveLayerFeature(self, image, layer):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', caffe.io.load_image(image))
        out = self.net.forward()
        items = {k:v.data for k, v in self.net.blobs.items()}
        return items[layer][0]

    def saveFeatures(self, root_folder, f_imagelist, layer, of):
        f_name = open(f_imagelist)
        cnt = 0
        featf = open(of + '.txt', 'w')
        try:
            for image in f_name.readlines():
                img = os.path.join(root_folder, image.strip().split(' ')[0])
                feature = self.__saveLayerFeature(img, layer)
                for v in feature:
                    featf.write(' ')
                    featf.write(str(v))
                featf.write('\n')
                cnt += 1
                if cnt % 100 == 0:
                    ISOTIMEFORMAT = '%Y-%m-%d %X'
                    print time.strftime(ISOTIMEFORMAT, time.localtime()), "%d tasks have done" % cnt
                    '''
                    print "--------------"
                    for tt in features:
                        print sum(tt)
                    print "--------------"
                    '''
        except Exception, e:
            print 'some error occured when processed', image
            print e
        f_name.close()
        featf.close()

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print ''' python extract_features.py vgg caffemodel root_folder imagelist fc7 [file_name]'''
        exit()

    cnn = CNN(sys.argv[1], sys.argv[2])
    cnn.saveFeatures(sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
