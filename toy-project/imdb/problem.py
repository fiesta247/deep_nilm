from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import imdb
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

@registry.register_problem
class MySentimentIMDB(imdb.SentimentIMDB):
    
    def _make_constant_shape(self, x, size):
        x = x[:size]
        xlen = tf.shape(x)[0]
        x = tf.pad(x, [[0, size - xlen]])
        return tf.reshape(x, [size])

    def preprocess_example(self, example, unused_mode, unused_hparams):
        example = super(MySentimentIMDB, self).preprocess_example(
            example, unused_mode, unused_hparams)
        example['inputs'] = self._make_constant_shape(example['inputs'], 2000)
        return example

