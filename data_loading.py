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

    class FaceLandmarksDataset(Dataset):
        """Face Landmarks dataset."""

        def __init__(self, csv_file, root_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a sample.
            """
            self.landmarks_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.landmarks_frame)

        def __getitem__(self, idx):
            image_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
            image = io.imread(image_name)
            landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
            landmarks = landmarks.astype("float").reshape(-1,2)
            sample = {"image":image, "landmarks":landmarks}
            if self.transform:
                sample = self.transform(sample)
            return sample


    # Let’s instantiate this class and iterate through the data samples.
    # We will print the sizes of first 4 samples and show their landmarks
    face_dataset = FaceLandmarksDataset(csv_file='faces/faces/face_landmarks.csv',
                                        root_dir='faces/faces/')
    fig = plt.figure()
    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape, sample['landmarks'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)

        if i == 3:
            plt.show()
            break


    # One issue we can see from the above is that the samples are not of the same size.
    # Most neural networks expect the images of a fixed size.
    # Therefore, we will need to write some prepocessing code. Let’s create three transforms:
    # Rescale: to scale the image
    # RandomCrop: to crop from image randomly. This is data augmentation.
    # ToTensor: to convert the numpy images to torch images (we need to swap axes).

    # Observe below how these transforms had to be applied both on the image and landmarks.

    class Rescale(object):
        """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """

        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

        def __call__(self, sample):
            image, landmarks = sample["image"], sample['landmarks']
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image=image, output_shape=(new_h, new_w))
            landmarks = landmarks * [new_w/w, new_h/h]

            return {"image":img, "landmarks":landmarks}





