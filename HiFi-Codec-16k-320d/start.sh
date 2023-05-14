
#sleep 666666666666666666666666666666666666666666666666666666666666666

set -e

proj_dir="path_of_academiCodec"

export PYTHONPATH=${proj_dir}:$PYTHONPATH
log_root="logs"

input_training_file="train.lst" # .lst save the wav path.
input_validation_file="valid_256.lst"

#mode=$1  # debug or train
#mode=debug
mode=train

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  export CUDA_VISIBLE_DEVICES=0
  python ${proj_dir}/train.py \
    --config ${proj_dir}/configs/param_config.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 100 \
    --summary_interval 10 \
    --validation_interval 100 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  python ${proj_dir}/train.py \
    --config ${proj_dir}/configs/param_config.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 5000
fi
