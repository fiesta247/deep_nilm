from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf

@registry.register_problem
class MySentimentIMDB(text_problems.Text2ClassProblem):
    """IMDB sentiment classification."""
    URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    
    """A single call to `generate_samples` generates for all `dataset_splits`.

    Set to True if you already have distinct subsets of data for each dataset
    split specified in `self.dataset_splits`. `self.generate_samples` will be
    called once for each split.

    Set to False if you have a unified dataset that you'd like to have split out
    into training and evaluation data automatically. `self.generate_samples`
    will be called only once and the data will be sharded across the dataset
    splits specified in `self.dataset_splits`.

    Returns:
      bool
    """
    @property
    def is_generate_per_split(self):
        return True
    
    """Splits of data to produce and number of output shards for each."""
    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 10,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
    @property
    def approx_vocab_size(self):
        return 2**13 # 8k vocab suffices for this small dataset.

    @property
    def num_classes(self):
        return 2

    def class_labels(self, data_dir):
        del data_dir
        return ["neg", "pos"]

    def doc_generator(self, imdb_dir, dataset, include_label=False):
        dirs = [(os.path.join(imdb_dir, dataset, "pos"), True), (os.path.join(
            imdb_dir, dataset, "neg"), False)]

        for d, label in dirs:
            for filename in os.listdir(d):
                with tf.gfile.Open(os.path.join(d, filename)) as imdb_f:
                    doc = imdb_f.read().strip()
                    if include_label:
                        yield doc, label
                    else:
                        yield doc
                        
    """Generate samples of text and label pairs.

    Each yielded dict will be a single example. The inputs should be raw text.
    The label should be an int in [0, self.num_classes).

    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).

    Yields:
      {"inputs": text, "label": int}
    """
    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Generate examples."""
        # Download and extract
        compressed_filename = os.path.basename(self.URL)
        download_path = generator_utils.maybe_download(tmp_dir,
                                                       compressed_filename,
                                                       self.URL)
        imdb_dir = os.path.join(tmp_dir, "aclImdb")
        if not tf.gfile.Exists(imdb_dir):
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(tmp_dir)

        # Generate examples
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset = "train" if train else "test"
        for doc, label in self.doc_generator(imdb_dir, dataset,
                                             include_label=True):
            yield {
                "inputs": doc,
                "label" : int(label),
            }

@registry.register_problem
class MySentimentIMDBCharacters(MySentimentIMDB):
    """IMDB sentiment classification, character level."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def global_task_id(self):
        return problem.TaskID.EN_CHR_SENT
