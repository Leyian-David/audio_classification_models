
import tensorflow as tf

from ..utils import math_util, shape_util
from .capsulenet import CapsuleLayer, PrimaryCap, Length, Mask


class TimeReduction(tf.keras.layers.Layer):
    def __init__(
        self,
        factor: int,
        name: str = "TimeReduction",
        **kwargs,
    ):
        super(TimeReduction, self).__init__(name=name, **kwargs)
        self.time_reduction_factor = factor

    def padding(
        self,
        time,
    ):
        new_time = tf.math.ceil(time / self.time_reduction_factor) * self.time_reduction_factor
        return tf.cast(new_time, dtype=tf.int32) - time

    def call(
        self,
        inputs,
        **kwargs,
    ):
        shape = shape_util.shape_list(inputs)
        outputs = tf.pad(inputs, [[0, 0], [0, self.padding(shape[1])], [0, 0]])
        outputs = tf.reshape(outputs, [shape[0], -1, shape[-1] * self.time_reduction_factor])
        return outputs

    def get_config(self):
        config = super(TimeReduction, self).get_config()
        config.update({"factor": self.time_reduction_factor})
        return config


class VggSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: tuple or list = (32, 64),
        kernel_size: int or list or tuple = 3,
        strides: int or list or tuple = 2,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="VggSubsampling",
        **kwargs,
    ):
        super(VggSubsampling, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            name=f"{name}_conv_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters[0],
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            name=f"{name}_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size=strides, padding="same", name=f"{name}_maxpool_1")
        self.conv3 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            name=f"{name}_conv_3",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=filters[1],
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            name=f"{name}_conv_4",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size=strides, padding="same", name=f"{name}_maxpool_2")
        self.time_reduction_factor = self.maxpool1.pool_size[0] * self.maxpool2.pool_size[0]

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.conv1(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool1(outputs, training=training)

        outputs = self.conv3(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv4(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.maxpool2(outputs, training=training)

        return math_util.merge_two_last_dims(outputs)

    def get_config(
        self,
    ):
        conf = super(VggSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.maxpool1.get_config())
        conf.update(self.conv3.get_config())
        conf.update(self.conv4.get_config())
        conf.update(self.maxpool2.get_config())
        return conf


class Conv2dSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        strides: list or tuple or int = 2,
        kernel_size: int or list or tuple = 3,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="Conv2dSubsampling",
        **kwargs,
    ):
        super(Conv2dSubsampling, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name=f"{name}_1",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            name=f"{name}_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.time_reduction_factor = self.conv1.strides[0] * self.conv2.strides[0]

    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        outputs = self.conv1(inputs, training=training)
        outputs = tf.nn.relu(outputs)
        outputs = self.conv2(outputs, training=training)
        outputs = tf.nn.relu(outputs)
        return math_util.merge_two_last_dims(outputs)

    def get_config(self):
        conf = super(Conv2dSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        return conf
    
class CaspNetSubsampling(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int = 256,
        strides: list or tuple or int = 1,
        padding: str = "valid",
        kernel_size: int or list or tuple = 9,
        kernel_regularizer=None,
        bias_regularizer=None,
        name="CaspNetSubsampling",
#         conv1_params = {
#             "filters": 256,
#             "kernel_size": 9,
#             "strides": strides,
#             "padding": padding,
#             "activation": "relu",
#             "kernel_regularizer": kernel_regularizer,
#             "bias_regularizer": bias_regularizer,
#             "name": f"{name}_conv1",
#         },
        **kwargs,
    ):
        super(CaspNetSubsampling, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="valid",
            name=f"{name}_conv1",
            activation="relu",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
#         self.conv1 = tf.keras.layers.Conv2D(**conv1_params)
        
    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        
        # Layer 1: Just a conventional Conv2D layer
        outputs = self.conv1(inputs, training=training)
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        primarycaps = PrimaryCap(outputs, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
        # Layer 3: Capsule layer. Routing algorithm works here.
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
        return digitcaps
    
    def get_config(self):
        conf = super(CaspNetSubsampling, self).get_config()
        conf.update(self.conv1.get_config())
        return conf
        
