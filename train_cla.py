import datetime
from gc import callbacks

from loaders.data_loader import Data_Seq
from models.pointnet_cla import Pointnet_Cla

import tensorflow as tf

if __name__ == "__main__":

#32個取り出して学習させる。全データを20回学習させる。1エポック(丸々1データセット学習)するにはバッチ*10回する必要がある。
    num_point = 2000 #点群の数　2000に揃えた。
    batch_size = 32
    epochs = 20
    ite_size = 10
    #train時のデータ
    train_seq = Data_Seq("../dataset_pointnet/polygon", num_point, batch_size, ite_size)
    # train_seq = Data_Seq("./dataset/trimesh_primitives/train", num_point, batch_size, ite_size)
    #test時のデータ
    val_seq = Data_Seq("../dataset_pointnet/polygon", num_point, batch_size, 1)
    # val_seq = Data_Seq("./dataset/trimesh_primitives/val", num_point, batch_size, 1)

    pointnet_cla = Pointnet_Cla(num_point, 16) #引数16はいくつ対象があるか。
    pointnet_cla.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    pointnet_cla.fit(
        x=train_seq,
        batch_size=batch_size,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = "./logs/weights-{}.h5".format(ts)
    print("save weights as :", save_path)
    pointnet_cla.save_weights(save_path)
