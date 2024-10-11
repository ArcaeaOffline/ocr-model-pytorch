#!/bin/bash
docker run \
    --rm \
    --gpus=all \
    --ipc=host \
    --volume .:/workspace \
    --volume ./docker/cache:/root/.cache \
    -p 6006:6006 \
    -it ArcaeaOffline/ocr-model-pytorch-trainer \
    /bin/bash
