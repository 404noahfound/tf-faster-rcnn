#!/bin/bash
GPU_ID='2,3'
CUDA_VISIBLE_DEVICES=${GPU_ID}
./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} coco_fpn res50 fpn multi_scale_anchor 
