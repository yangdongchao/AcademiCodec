## How to train your model
Firstly, set the related path in start.sh file, then <br>
`
bash start.sh
`

## How to Inference
set the proj_dir as same as training process. Then <br>
`
bash test.sh
`

## How to use pre-trained model
If you donot want to train the model, and directly using our pre-trained models. Firstly, please set the model from: huggingface https://huggingface.co/Dongchao/AcademiCodec/tree/main  <br>
Then using following command:
`
CUDA_VISIBLE_DEVICES=0 python ./vqvae_copy_syn.py \
    --model_path your_ckpts_path \
    --config_path config16k_320d.json \
    --input_wavdir your wave folder \
    --outputdir your output folder \
    --num_gens 10000
`