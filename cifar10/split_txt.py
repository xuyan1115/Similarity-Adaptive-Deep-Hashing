#! /usr/bin/env python
# -*- coding: utf-8 -*-

f = open('./cifar10_train_flip.txt', 'r')
f2 = open('./cifar10_train_flip1.txt', 'w')

while True:
    temp = f.readline().strip()
    if temp == "":
        break
    else:

        f2.write(temp + ' 0\n')

f.close()
f2.close()
