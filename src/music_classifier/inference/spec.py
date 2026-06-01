import tensorflow as tf
import keras
from keras import layers

@keras.saving.register_keras_serializable()
class SpecAugment(layers.Layer):
    """
    SpecAugmentation that removes parts of frequency bands to
    help fight overfitting.

    When the model is loaded Keras needs to be able to access this custom 
    class as it is part of the model. However, this code does not run at 
    inference because Training is False. The inputs just run through. 
    """
    def __init__(self, freq_mask_param=15, time_mask_param=15, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def call(self, x, training=None):
        if not training:
            return x   # no-op at inference time

        shape = tf.shape(x)
        batch, time_steps, freq_bins, channels = shape[0], shape[1], shape[2], shape[3]

        # ── Frequency mask ───────────────────────────────────────────────────
        f  = tf.random.uniform((), 0, self.freq_mask_param, dtype=tf.int32)
        f0 = tf.random.uniform((), 0, freq_bins - f,        dtype=tf.int32)
        freq_mask = tf.concat([
            tf.ones ([batch, time_steps, f0,             channels]),
            tf.zeros([batch, time_steps, f,              channels]),
            tf.ones ([batch, time_steps, freq_bins-f0-f, channels])
        ], axis=2)
        x = x * freq_mask

        # ── Time mask ────────────────────────────────────────────────────────
        t  = tf.random.uniform((), 0, self.time_mask_param, dtype=tf.int32)
        t0 = tf.random.uniform((), 0, time_steps - t,       dtype=tf.int32)
        time_mask = tf.concat([
            tf.ones ([batch, t0,              freq_bins, channels]),
            tf.zeros([batch, t,               freq_bins, channels]),
            tf.ones ([batch, time_steps-t0-t, freq_bins, channels])
        ], axis=1)
        x = x * time_mask

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"freq_mask_param": self.freq_mask_param,
                        "time_mask_param": self.time_mask_param})
        return config