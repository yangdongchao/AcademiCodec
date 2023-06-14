#!/bin/bash
source path.sh

python3 ${BIN_DIR}/main3_ddp.py \
        --BATCH_SIZE 16 \
        --N_EPOCHS 300 \
        --save_dir path_to_save_log \
        --PATH  path_to_save_model \
        --train_data_path path_to_training_data \
        --valid_data_path path_to_val_data \
        --ratios 8 5 4 2 \
        --sr 16000