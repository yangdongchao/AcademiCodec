# The training code of Encodec

### Note that, this part of code is based on Facebook's Encodec. We just provide the training process. The license is the same as Encodec.

### For Training
set the right path to statr/start.sh

run: bash start.sh

### For Inference
if you want to use our checkpoint. Run the following 
mkdir checkpoint
cd checkpoint
wget https://huggingface.co/Dongchao/AcademiCodec/resolve/main/encodec_16khz_320d.pth
bash test.sh    # set the root in test.sh, before runing it.