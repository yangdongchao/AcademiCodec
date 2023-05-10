# AcademiCodec: An Open Source Audio Codec Model for Academic Research

### On going
This project is on going. You can find the paper on https://arxiv.org/pdf/2305.02765.pdf <br/>
Furthermore, this project is lanched from University, we expect more researchers to be the contributor. <br/>

#### Abstract <wip>
Audio codec models are widely used in audio communication as a crucial technique for compressing audio into discrete representations. Nowadays, audio codec models are increasingly utilized in generation fields as intermediate representations. For instance, AudioLM is an audio generation model that uses the discrete representation of SoundStream as a training target, while VALL-E employs the Encodec model as an intermediate feature to aid TTS tasks. Despite their usefulness, two challenges persist: (1) training these audio codec models can be difficult due to the lack of publicly available training processes and the need for large-scale data and GPUs; (2) achieving good reconstruction performance requires many codebooks, which increases the burden on generation models. In this study, we propose a group-residual vector quantization (GRVQ) technique and use it to develop a novel \textbf{Hi}gh \textbf{Fi}delity Audio Codec model, HiFi-Codec, which only requires 4 codebooks. We train all the models using publicly available TTS data such as LibriTTS, VCTK, AISHELL, and more, with a total duration of over 1000 hours, using 8 GPUs. Our experimental results show that HiFi-Codec outperforms Encodec in terms of reconstruction performance despite requiring only 4 codebooks. To facilitate research in audio codec and generation, we introduce AcademiCodec, the first open-source audio codec toolkit that offers training codes and pre-trained models for Encodec, SoundStream, and HiFi-Codec.

## 🔥 News
#### AcademiCodec
- 2023.4.16: We first release the training code for Encodec and SoundStream and our pre-trained models, includes 24khz and 16khz.
- 2023.5.5: We release the code of HiFi-Codec.

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
* Encodec_24k_240d, we train it using 24khz audio, and we set the downsample as 240, which can be used to InstructTTS
* Encodec_24k_32d, we train it using 24khz audio, we only set the downsample as 32, which can only use one codebook, such as AudioGen.
* SoundStream_24k_240d, the same configuration as Encodec_24k_240d.
## What the difference between SoundStream, Encodec and HiFi-Codec?
In our view, the mian difference between SoundStream and Encodec is the different Discriminator choice. For Encodec, it only uses a STFT-dicriminator, which forces the STFT-spectrogram be more real. SoundStream use two types of Discriminator, one forces the waveform-level to be more real, one forces the specrogram-level to be more real. In our code, we adopt the waveform-level discriminator from HIFI-GAN. The spectrogram level discrimimator from Encodec. In thoery, we think SoundStream enjoin better performance. Actually, Google's offical SoundStream proves this, Google can only use 3 codebooks to reconstruct a audio with high-quality. Although our implements can also use 3 codebooks to realize good performance, we admit our version cannot be compared with Google now. <br/>
For the HiFi-Codec, which is our proposed novel methods, which aims to help to some generation tasks. Such as VALL-E, AudioLM, MusicLM, SpearTTS, IntructTTS and so on. HiFi-Codec codebook only needs 4 codebooks, which significantly reduce the token numbers. Some researchers use our HiFi-Codec to implement VALL-E, which proves that can get better audio quality.

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
https://github.com/facebookresearch/encodec <br>
https://github.com/yangdongchao/Text-to-sound-Synthesis <br>
https://github.com/b04901014/MQTTS
## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{yang2023instructtts,
  title={InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt},
  author={Yang, Dongchao and Liu, Songxiang and Huang, Rongjie and Lei, Guangzhi and Weng, Chao and Meng, Helen and Yu, Dong},
  journal={arXiv preprint arXiv:2301.13662},
  year={2023}
}
```

## Disclaimer ##
Note that part of the code is based on Encodec, so that the license should be the same as Encodec. All of our code and pre-trained models can be only used for Academic research (non-commercial).

