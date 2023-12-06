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
dataset_path = "~/python/dataset_pointnet"
dirname = os.path.dirname(dataset_path)
meta_path = 'PartAnnotation/metadata.json'
with open( os.path.join(dirname, meta_path) ) as json_file:
  metadata = json.load(json_file)

target = 'Chair'
obj_dir = metadata[target]['directory']
points_dir = os.path.join(dirname, 'PartAnnotation', obj_dir, 'points')
points_files = glob(os.path.join(points_dir,"*.pts"))

#まずは1つのファイルから開く。
# coords = []
# for point_file in tqdm(points_files): #複数のファイルから、
#     # point_cloud = np.loadtxt(point_file)
#     with open(point_file,'r') as f:
#         pt_list = f.read().split('\n')
        
#     if pt_list.shape[0] < NUM_SAMPLE_POINTS:
#         continue
point_clouds_np = []
for point_file in tqdm(points_files):
  point_cloud_np = np.loadtxt(point_file)
  if point_cloud_np.shape[0] < NUM_SAMPLE_POINTS:
    continue
point_clouds = point_clouds_np.tolist()


    
data_to_save = {'coords':point_clouds}

file_name = 'dataset/pointcloud/sample.json'
with open(file_name,'w') as file:
    json.dump(data_to_save,file)