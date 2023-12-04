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

# shapenetデータセットの一部であるPASCAL 3D+をダウンロード
dataset_url = "https://media.githubusercontent.com/media/kaz12tech/datasets/main/shapenet.zip?download=true"

dataset_path = keras.utils.get_file(
    fname="shapenet.zip",
    origin=dataset_url,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=True,
    archive_format="auto",
    cache_dir="datasets",
)
print('dataset path:', dataset_path)
