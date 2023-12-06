import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pprint

# 乱数固定
tf.random.set_seed(12)

#こっちにあって良い。

#ptsファイルには行ごとに(x,y,z)が入っている。
#行ごとに読み込んで、それを{'coords':[0.03,0.02,-0.04]}のような形に変換したい。


NUM_SAMPLE_POINTS = 1024  # サンプリング数
dataset_path = "../dataset_pointnet"
# dirname = os.path.dirname(dataset_path) dirname = ~/pythonだった
# print("dirname",dirname)
meta_path = 'PartAnnotation/metadata.json'
with open( os.path.join(dataset_path, meta_path) ) as json_file:
  metadata = json.load(json_file)

target = 'Chair'
obj_dir = metadata[target]['directory']
print("obj_dir",obj_dir)
points_dir = os.path.join(dataset_path, 'PartAnnotation', obj_dir, 'points')
points_files = glob(os.path.join(points_dir,"*.pts"))

#まずは1つのファイルから開く。
# coords = []
# for point_file in tqdm(points_files): #複数のファイルから、
#     # point_cloud = np.loadtxt(point_file)
#     with open(point_file,'r') as f:
#         pt_list = f.read().split('\n')
        
#     if pt_list.shape[0] < NUM_SAMPLE_POINTS:
#         continue
point_clouds = []
for point_file in tqdm(points_files):
  point_clouds = np.loadtxt(point_file)
  if point_clouds.shape[0] < NUM_SAMPLE_POINTS:
    continue


    
data_to_save = {'coords':point_clouds}

file_name = os.path.join(dataset_path,'sample.json')
with open(file_name,'w') as file:
    json.dump(data_to_save,file)