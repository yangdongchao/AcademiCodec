移动了代码位置后 `test.sh` 可以跑通，`start.sh` 暂未验证
## How to train your model
Firstly, set the related path in start.sh file, then <br>
```bash
bash start.sh
```

## How to Inference
```bash
mkdir checkpoint
cd checkpoint
wget https://huggingface.co/Dongchao/AcademiCodec/resolve/main/HiFi-Codec-24k-240d
bash test.sh
```
