#!/bin/bash
GPU_ID='7'
CUDA_VISIBLE_DEVICES=${GPU_ID}
./experiments/scripts/test_faster_rcnn.sh ${GPU_ID} coco_fpn res50 fpn
