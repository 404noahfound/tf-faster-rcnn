#!/bin/bash
GPU_ID='2'
CUDA_VISIBLE_DEVICES=${GPU_ID}
./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} coco_minival res50
