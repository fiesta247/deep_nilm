#!/bin/bash

floyd run \
    --cpu \
    --env tensorflow-1.15 \
    --message "Generate IMDB dataset for Tensor2Tensor" \
    --follow \
    "sh toy_projects/imdb/create_dataset.sh"
