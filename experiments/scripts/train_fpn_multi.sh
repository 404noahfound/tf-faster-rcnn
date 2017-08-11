#!/bin/bash
GPU_ID='6,7'
CUDA_VISIBLE_DEVICES=${GPU_ID}
./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} coco_minival res50 fpn_multi
