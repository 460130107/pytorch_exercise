#!/usr/bin/env python  
# coding: utf-8  


""" 
@version: v1.0 
@author: Styang 
@license: Apache Licence  
@contact: 460130107@qq.com 
@site:  
@software: PyCharm Community Edition 
@file: data_loading.py 
@time: 2019/3/11 22:21 
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    plt.ion()

    landmarks_frame = pd.read_csv('faces/faces/face_landmarks.csv')
    n = 65
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))

    print(landmarks[:, 0])
    print(landmarks[:, 1])

    def show_landmarks(image, landmarks):
        """Show image with landmarks"""
        plt.imshow(image)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
        plt.pause(0.001)  # pause a bit so that plots are updated


    plt.figure()
    show_landmarks(io.imread(os.path.join('faces/faces/', img_name)), landmarks)
    plt.show()