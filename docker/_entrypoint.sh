#!/bin/bash
nohup tensorboard --logdir logs/tensorboard --bind_all &
python train.py
exit 0
