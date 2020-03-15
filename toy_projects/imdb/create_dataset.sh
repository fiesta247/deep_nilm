#!/bin/bash

readonly PROBLEM=sentiment_imdb

readonly TMP_DIR=$(mktemp -d)
DATA_DIR="datasets/imdb"

CMD="t2t-datagen"
CMD="${CMD} --data_dir=${DATA_DIR}"
CMD="${CMD} --tmp_dir=${TMP_DIR}"
CMD="${CMD} --problem=${PROBLEM}"

${CMD}
