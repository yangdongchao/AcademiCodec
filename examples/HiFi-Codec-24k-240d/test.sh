
proj_dir="path_of_proj" # 
log_root="logs"
ckpt="$(ls -dt "${log_root}"/g_* | head -1 || true)"
echo checkpoint path: ${ckpt}

# the path of test wave
wav_dir="test_wavs"

outputdir=${log_root}/copysyn_$(date '+%Y-%m-%d-%H-%M-%S')
mkdir -p ${outputdir}

CUDA_VISIBLE_DEVICES=0 python ./vqvae_copy_syn.py \
    --model_path ${ckpt} \
    --config_path ${log_root}/config.json \
    --input_wavdir ${wav_dir} \
    --outputdir ${outputdir} \
    --num_gens 10000
