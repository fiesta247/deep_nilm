"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.models import basic

import tensorflow.compat.v1 as tf

@registry.register_model
class MyFC(t2t_model.T2TModel):
    """Basic fully-connected + ReLU model."""
    
    def body(self, features):
        hparams = self.hparams
        x = features["inputs"]
        shape = common_layers.shape_list(x)
        tf.logging.info(shape)
        x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
        for i in range(hparams.num_hidden_layers):
            x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
            x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
            x = tf.nn.relu(x)
        return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.

@registry.register_hparams
def my_hparams():
    """Small fully connected model."""
    hparams = basic.basic_fc_small()
    return hparams

