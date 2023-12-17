import os
import json
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

object_dir_path = '../dataset_pointnet_normalized/pointcloud'
object_names = os.listdir(object_dir_path)
pointcloud = []
train_dir = '../dataset_pointnet_normalized/pc-split/train' 
test_dir =  '../dataset_pointnet_normalized/pc-split/test'

#それぞれのdirectory(Table,Cap)で分割したい

for object_name in tqdm(object_names): #Tableなど
    object_path = os.path.join(object_dir_path,object_name)
    ids = os.listdir(object_path)
    
    #もしsave_dirの先が作られいなければ作成しておく
    if not os.path.exists(os.path.join(train_dir,object_name)):
        os.makedirs(os.path.join(train_dir,object_name))
        
    if not os.path.exists(os.path.join(test_dir,object_name)):
        os.makedirs(os.path.join(test_dir,object_name))
    
    
        
    #train,testに分割する　-> (i)directory名で保存しておく (ii)ファイルごと保存しておく。
    #ファイルで保存しておくのは重くなりすぎる。directoryで保存しておいて順にコピーしていく。
    
    train_data, test_data = train_test_split(ids,test_size=0.1,random_state=42)
    
    #train_data,test_dataには分割したファイル名が保存されている。
    #train,testに分割したデータをそれぞれコピーする
    #train_dataの中のすべてのデータに対してコピーを行う。forを用いる。
    
    for a_train_data in train_data:
        #copy後のdirectory
        save_train_dir=os.path.join(train_dir,object_name,a_train_data)
        #copy前のdirectory
        initial_dir = os.path.join(object_path,a_train_data)
        shutil.copy(initial_dir,save_train_dir)
        
    
    for a_test_data in test_data:
        #copy後のdirectory
        save_test_dir=os.path.join(test_dir,object_name,a_test_data)
        #copy前のdirectory
        initial_dir = os.path.join(object_path,a_test_data)
        shutil.copy(initial_dir,save_train_dir)
        



