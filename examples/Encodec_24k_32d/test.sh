#!/bin/bash

python3 test.py \
       --input=./test_wav \
       --output=./output \
       --resume_path=checkpoint/encodec_16k_320d.pth \
       --sr=16000
       