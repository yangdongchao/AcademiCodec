#!/bin/bash
source path.sh

python3 test.py \
       --input=./test_wav \
       --output=./output \
       --resume_path=checkpoint/Encodec_24khz_32d.pth \
       --sr=24000
       