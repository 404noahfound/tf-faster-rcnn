#!/bin/bash
GPU_ID='1,2,3'
CUDA_VISIBLE_DEVICES=${GPU_ID}
./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} coco_minival res50 fpn
