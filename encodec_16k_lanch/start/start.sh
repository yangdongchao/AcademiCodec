
#sleep 888888888888888888888888888888888888888888888888888888888

cd /code/soundstream_16k_lanch


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m torch.distributed.launch --nproc_per_node 8 main_launch.py \
  --save_dir log \
  --PATH model_path \
  --BATCH_SIZE 24  --N_EPOCHS 2000
