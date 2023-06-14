#!/bin/bash
source path.sh

python3 ${BIN_DIR}/main3_ddp.py \
        --BATCH_SIZE 16 \
        --N_EPOCHS 300 \
        --save_dir path_to_save_log \
        --PATH  path_to_save_model \
        --train_data_path path_to_training_data \
        --valid_data_path path_to_val_data \
        --sr 24000 \
        --ratios 2 2 2 4 \
        --target_bandwidths 7.5 15
