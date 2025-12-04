from dataManagement.DatasetSplit import DatasetSplit
import os
import json
import chardet
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt

# 分离训练集和验证集
def train_val_split(data, val_size=0.2):
    return train_test_split(data, train_size=(1-val_size), test_size=val_size)

class DatasetLoader(object):

    def __init__(self):
        self.train = DatasetSplit()
        self.val = DatasetSplit()
        self.test = DatasetSplit()

    def load_data(self, train_path,test=False):
        print('Loading data...')
        
        data = []
        df = pd.read_csv(train_path)
        print('----- [Loading]')
        for index, row in df.iterrows():
            image_path, describe, label = row['image'], row['describe'], row['label']
            data.append((image_path, describe, label))
        if test == False:
            train_data, val_data = train_val_split(data)
            self.train.load_data(train_data)
            train_labels = self.train.get_labels()
            train_images = self.train.get_images()
            train_texts = self.train.get_texts()

            self.val.load_data(val_data)
            val_labels = self.val.get_labels()
            val_images = self.val.get_images()
            val_texts = self.val.get_texts()

            print('Train/val split: {:d}/{:d}'.format(len(train_texts), len(val_texts)))

            self.set_train_data(train_labels, train_images, train_texts)
            self.set_val_data(val_labels, val_images, val_texts)
        else:
            self.test.load_data(data)
            test_labels = self.test.get_labels()
            test_images = self.test.get_images()
            test_texts = self.test.get_texts()

    def set_train_data(self, train_labels, train_images, train_texts):
        self.train.set_labels(train_labels)
        self.train.set_images(train_images)
        self.train.set_texts(train_texts)

    def set_val_data(self, val_labels, val_images, val_texts):
        self.val.set_labels(val_labels)
        self.val.set_images(val_images)
        self.val.set_texts(val_texts)
    
    def set_test_data(self, val_labels, val_images, val_texts):
        self.test.set_labels(val_labels)
        self.test.set_images(val_images)
        self.test.set_texts(val_texts)

    def get_train_data(self):
        return self.train

    def get_val_data(self):
        return self.val
    
    def get_test_data(self):
        return self.test
