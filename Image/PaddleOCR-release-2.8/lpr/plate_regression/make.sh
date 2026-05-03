#!/usr/bin/env bash
cd ./ssd_detector/utils

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace

cd ..