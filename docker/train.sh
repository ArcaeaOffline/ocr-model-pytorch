#!/bin/bash
# --ipc=host: avoid `insufficient shared memory (shm)` errors
# /root/.cache: matplotlib and torch hub caches
docker run \
    --rm \
    --gpus=all \
    --ipc=host \
    --volume .:/workspace \
    --volume ./docker/cache:/root/.cache \
    -p 6006:6006 \
    ArcaeaOffline/ocr-model-pytorch-trainer \
    /bin/bash ./docker/_entrypoint.sh
