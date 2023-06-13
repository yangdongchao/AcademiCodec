#!/bin/bash
source path.sh

python3 test.py \
       --input=./test_wav \
       --output=./output \
       --resume_path=checkpoint/encodec_24khz_240d.pth \
       --sr=24000
       