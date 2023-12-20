import tensorflow as tf
from loaders.data_loader import Data_Seq
from models.pointnet_cla import Pointnet_Cla

# ハイパーパラメータ
num_point = 2000
batch_size = 32

# テストデータの読み込み
test_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud_3Dmodel", num_point, 1, 10)

# モデルの構築
pointnet_cla = Pointnet_Cla(num_point, 16)

# モデルの重みを読み込む（前提）
pointnet_cla.load_weights("./logs/weights-20231220-205259.h5")

# オプティマイザと損失関数の設定
# pointnet_cla.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


print("aaaaa",test_seq)
#(x_train,y_train,verbose=0
test_loss,test_acc = pointnet_cla.evaluate(x=test_seq)
print("test_loss,test_acc:",test_loss,test_acc)
