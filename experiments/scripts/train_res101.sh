#!/bin/bash
GPU_ID='0,1,2,3'
CUDA_VISIBLE_DEVICES=${GPU_ID} 
./experiments/scripts/train_faster_rcnn.sh ${GPU_ID} pascal_voc res101
