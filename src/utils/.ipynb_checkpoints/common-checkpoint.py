'''
普通的常用工具

'''

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

# 写入数据
def write_to_file(path, outputs, save_path=None):
    if save_path == None:
        save_path = path
    df = pd.read_csv(path)
    df['predict'] = outputs
    df.to_csv(save_path,index=False)

# 保存模型
def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)    # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)