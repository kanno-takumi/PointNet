import tensorflow as tf
from loaders.data_loader import Data_Seq
from models.pointnet_cla import Pointnet_Cla

# ハイパーパラメータ
num_point = 2000
batch_size = 32

# テストデータの読み込み
test_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud_3Dmodel", num_point, batch_size, 1)

# モデルの構築
pointnet_cla = Pointnet_Cla(num_point, 16)

# モデルの重みを読み込む（前提）
pointnet_cla.load_weights("./logs/weights-20231220-133758.h5")

# オプティマイザと損失関数の設定
# pointnet_cla.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

pointnet_cla.evaluate(x=test_seq)
