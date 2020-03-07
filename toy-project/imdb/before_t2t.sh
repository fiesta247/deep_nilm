#!/bin/bash

if [ -z "$ROOTDIR" ]
then
	readonly ROOTDIR=$(git rev-parse --show-toplevel)
fi

echo -e "\nt2t variables\n"

PROBLEM=my_sentiment_imdb 
echo "problem: $PROBLEM"

MODEL=my_fc
echo "model: $MODEL"

HPARAMS=my_hparams
echo "hparams: $HPARAMS"

# Directories
echo -e "\nDirectories\n"

USR_DIR=$ROOTDIR/toy-project/imdb
echo "usr_dir: $USR_DIR"

DATA_DIR=$ROOTDIR/t2t/data
echo "data_dir: $DATA_DIR"

TMP_DIR=$ROOTDIR/t2t/tmp/datagen
echo "tmp_dir: $TMP_DIR"

TRAIN_DIR=$ROOTDIR/t2t/train/$PROBLEM/$MODEL-$HPARAMS
echo "train_dir: $TRAIN_DIR"
