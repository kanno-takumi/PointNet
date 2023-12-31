import datetime
from gc import callbacks

from loaders.data_loader import Data_Seq
from models.pointnet_cla import Pointnet_Cla

import tensorflow as tf

if __name__ == "__main__":

#32個取り出して学習させる。全データを20回学習させる。1エポック(丸々1データセット学習)するにはバッチ*10回する必要がある。
    # train_file_num =31384
    # test_file_num = 57
    
    num_point = 2000 #点群の数　2000に揃えた。
    batch_size = 16
    epochs = 5
    ite_size = 10
    # train_ite_size = int(train_file_num/batch_size)
    # print(train_ite_size)
    
    #train時のデータ
    # train_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud-3", num_point, batch_size, ite_size)#train_ite_size
    
    # train_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud", num_point, batch_size, 980)#train_ite_size
    train_seq = Data_Seq("../dataset_pointnet_normalized/pc-split/train3label", num_point, batch_size, ite_size)
    # train_seq = Data_Seq("./dataset/trimesh_primitives/train", num_point, batch_size, ite_size)
    #test時のデータ
    val_seq = Data_Seq("../dataset_pointnet_normalized/pc-split/test3label", num_point, batch_size, 1)
    # val_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud_3Dmodel", num_point, batch_size, 1)
    # val_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud_3Dmodel-3", num_point, 3, 1)
    # val_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud_3Dmodel", num_point, 1, 1)
    # val_seq = Data_Seq("./dataset/trimesh_primitives/val", num_point, batch_size, 1)

    pointnet_cla = Pointnet_Cla(num_point, 3) #引数16はいくつ対象があるか。
    pointnet_cla.summary()#modelを表示する？

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    #訓練の実行 tensorflow.keras
    pointnet_cla.fit(
        x=train_seq,
        batch_size=batch_size,
        validation_data=val_seq, #訓練と評価を別で行う
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )
    
    test_loss, test_acc = pointnet_cla.evaluate(train_seq,  val_seq, verbose=2) #verbose:ログ出力モード
    print("test_loss:",test_loss)
    print("test_acc:",test_acc)
    predictions = pointnet_cla.predict(val_seq)
    print("predictions:",predictions)

    #評価
    # pointnet_cla.evaluate(x=val_seq)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./logs/weights-{}.h5".format(ts)
    print("save weights as :", save_path)
    pointnet_cla.save_weights(save_path)
