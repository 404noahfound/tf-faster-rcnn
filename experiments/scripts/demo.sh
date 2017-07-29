#!/bin/bash

GPU_ID=4,5,6
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py
