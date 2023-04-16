# AcademiCodec: An Open Source Audio Codec Model for Academic Research

### On going
This project is on going. We will release the our technical reports and all of the code as soon as. <br/>
Furthermore, this project is lanched from University, we expect more researchers to be the contributor. <br/>

#### Abstract <wip>
Audio codec models are used to compress audio into a group of discrete representations, which is an critic technique in audio communication. Nowadays, audio codec models are widely employed on generation fields, which is used as intermediate representations. For example, a audio generation model, AudioLM, utilizes the discrete representation of SoundStream as training target. VALL-E uses Encodec model as intermediate features to help finish TTS tasks. Although SoundStream and Encodec are very useful codec models, training such a model is difficult in academic fields. The main reason is that the training process is not public accessible, and codec needs large-scale data and GPUs. Most of academic researchers can only use their pre-trained models for their research. However, it is not convenient when we need different configurations (such as sample rate, down-sample times). In this study, we provide a open source codec model for academic research. We first provide our training process code for SoundStream and Encodec. Then we present our proposed group-residual vector quantization (GRVQ) technique. Based on GRVQ technique, we build a novel codec model, AcademiCodec. We train all of the models on public available TTS data, such as LibriTTS, VCTK, AISHELL and so on, the total duration is more than 1000 hours. We finish all of training process based on 8 NVIDIA 3090 GPUs, which means that almost all of researcher can train their own codec models. Furthermore, we will discuss the critic training steps in audio codec, such as how to choose discriminator, how to balance the loss function. 
We provide our implementation and pretrained models in this repository.

## 🔥 News
#### AcademiCodec
- 2023.4.16: We first release the training code for Encodec and SoundStream and our pre-trained models, includes 24khz and 16khz.

### Dependencies
* [PyTorch](http://pytorch.org/) version >= 1.13.0
* Python version >= 3.8

# Train your own model
  please refer to the specific version.

## Data preparation
Just prepare your audio data in one folder. Make sure the sample rate is right.

## Training or Inferce
Refer to the specical folders, e.g. Encodec_24k_240d represent, the Encodec model, sample rate is 24khz, downsample rate is 240. If you want to use our pre-trained models, please refer to https://huggingface.co/Dongchao/AcademiCodec/tree/main

## Version Description
* Encodec_16k_320, we train it using 16khz audio, and we set the downsample as 320, which can be used to train SpearTTS
* Encodec_24k_240d, we train it using 24khz audio, and we set the downsample as 320, which can be used to InstructTTS
* Encodec_24k_32d, we train it using 24khz audio, we only set the downsample as 32, which can only use one codebook, such as AudioGen.
* SoundStream_24k_240d, the same configuration as Encodec_24k_240d.
## What the difference of SoundStream, Encodec and AcademiCodec?
In our view, the mian difference between SoundStream and Encodec is the different Discriminator choice. For Encodec, it only uses a STFT-dicriminator, which forces the STFT-spectrogram be more real. SoundStream use two types of Discriminator, one forces the waveform-level to be more real, one forces the specrogram-level to be more real. In our code, we adopt the waveform-level discriminator from HIFI-GAN. The spectrogram level discrimimator from Encodec. In thoery, we think SoundStream enjoin better performance. Actually, Google's offical SoundStream proves this, Google can only use 3 codebooks to reconstruct a audio with high-quality. Although our implements can also use 3 codebooks to realize good performance, we admit our version cannot be compared with Google now. <br/>
For the AcademiCodec, which is our proposed novel methods, which aims to help to some generation tasks. Such as VALL-E, AudioLM, MusicLM, SpearTTS, IntructTTS and so on. Academic codebook only needs 4 codebooks, which significantly reduce the token numbers. Some research use our AcademiCodec to implement VALL-E, which proves that can get better audio quality.

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
https://github.com/facebookresearch/encodec

## Citations ##
If you find this code useful in your research, please cite our work:

## Disclaimer ##
Note that part of the code is based on Encodec, so that the license should be the same as Encodec. All of our code and pre-trained models can be only used for Academic research (non-commercial).

