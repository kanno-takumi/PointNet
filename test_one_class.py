import tensorflow as tf
from loaders.data_loader import Data_Seq
from models.pointnet_cla import Pointnet_Cla

def test_one_class(model, test_data):
    test_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for test_batch in test_data:
        predictions = model(test_batch)
        loss = model.evaluate(test_batch, predictions)

        test_loss += loss[0]

        # 一つのラベルに対する予測と正解の比較
        correct_predictions += tf.reduce_sum(tf.cast(tf.math.argmax(predictions, axis=1) == test_batch, tf.int32)).numpy()
        total_samples += test_batch.shape[0]

    average_test_loss = test_loss / total_samples
    accuracy = correct_predictions / total_samples

    print(f"Test Loss: {average_test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

# ハイパーパラメータ
num_point = 2000
batch_size = 32

# テストデータの読み込み
test_seq = Data_Seq("../dataset_pointnet_normalized/pointcloud_3Dmodel", num_point, batch_size, 1)

# モデルの構築
pointnet_cla = Pointnet_Cla(num_point, 16)

# オプティマイザと損失関数の設定
# pointnet_cla.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# モデルの重みを読み込む（前提）
pointnet_cla.load_weights("./logs/weights-20231220-133758.h5")

# テスト
test_one_class(pointnet_cla, test_seq)
