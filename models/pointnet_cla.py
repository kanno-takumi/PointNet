import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# https://keras.io/examples/vision/pointnet/

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

# def tnet(inputs, num_features):

#     # Initalise bias as the indentity matrix
#     bias = keras.initializers.Constant(np.eye(num_features).flatten())
#     reg = OrthogonalRegularizer(num_features)

#     x = conv_bn(inputs, 32)
#     x = conv_bn(x, 64)
#     x = conv_bn(x, 512)
#     x = layers.GlobalMaxPooling1D()(x)
#     x = dense_bn(x, 256)
#     x = dense_bn(x, 128)
#     x = layers.Dense(
#         num_features * num_features,
#         kernel_initializer="zeros",
#         bias_initializer=bias,
#         activity_regularizer=reg,
#     )(x)
#     feat_T = layers.Reshape((num_features, num_features))(x)
#     # Apply affine transformation to input features
#     return layers.Dot(axes=(2, 1))([inputs, feat_T])

class tnet(tf.keras.Model):
    """
    tnet model implementation
    """
    def __init__(self, num_points, num_features, name=""):
        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        inputs = keras.Input(shape=(num_points, num_features))
        x = conv_bn(inputs, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = dense_bn(x, 256)
        x = dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        outputs =  layers.Dot(axes=(2, 1))([inputs, feat_T])
        ### ...define leyers
        super(tnet, self).__init__(inputs=inputs, outputs=outputs, name=name)


class Pointnet_Cla(tf.keras.Model):
     def __init__(self, num_points, num_class):
        print("init Pointnet_Cla")
        ### define leyers
        inputs = keras.Input(shape=(num_points, 3))
        x = tnet(num_points, 3)(inputs)
        # x=tnet(inputs, 3)
        x = conv_bn(x, 32)
        x = conv_bn(x, 32)
        x = tnet(num_points, 32)(x)
        # x = tnet(x, 32)
        x = conv_bn(x, 32)
        x = conv_bn(x, 64)
        x = conv_bn(x, 512)
        glb_feature = layers.GlobalMaxPooling1D(name="glb_feature")(x)
        x = dense_bn(glb_feature, 256)
        x = layers.Dropout(0.3)(x)
        x = dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_class, activation="softmax")(x)
        ### ...define leyers
        super(Pointnet_Cla, self).__init__(inputs=inputs, outputs=outputs)

        adam = tf.optimizers.Adam(learning_rate=0.001, decay=0.0)
        loss = tf.losses.sparse_categorical_crossentropy
        metric = keras.metrics.sparse_categorical_accuracy
        self.compile(optimizer=adam, loss=loss , metrics=metric)

        
if __name__ == "__main__":
    pointnet_cla = Pointnet_Cla(128, 5)
    pointnet_cla.summary()
