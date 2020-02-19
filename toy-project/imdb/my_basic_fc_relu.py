
"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow.compat.v1 as tf

@registry.register_model
class MyBasicFcRelu(t2t_model.T2TModel):
    """Basic fully-connected + ReLU model."""
    def body(self, features):
        hparams = self.hparams
        x = features["inputs"]
        shape = common_layers.shape_list(x)
        tf.logging.info('------------------------ {}'.format(shape))
        x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
        for i in range(hparams.num_hidden_layers):
            x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
            x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
            x = tf.nn.relu(x)
        return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.

@registry.register_hparams
def my_basic_fc_small():
    """Small fully connected model."""
    hparams = common_hparams.basic_params1()
    hparams.learning_rate = 0.1
    hparams.batch_size = 128
    hparams.hidden_size = 256
    hparams.num_hidden_layers = 2
    hparams.initializer = "uniform_unit_scaling"
    hparams.initializer_gain = 1.0
    hparams.weight_decay = 0.0
    hparams.dropout = 0.0
    return hparams

