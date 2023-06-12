import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from librosa.util import normalize
from tqdm import tqdm

from vqvae import VQVAE


class VqvaeTester(nn.Module):

    def __init__(self, config_path, model_path, sample_rate=24000):
        super().__init__()
        self.vqvae = VQVAE(config_path, model_path, with_encoder=True)
        self.sample_rate = sample_rate

    @torch.no_grad()
    def forward(self, wav_path):
        wav, sr = sf.read(wav_path)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        fid = os.path.basename(wav_path)[:-4]
        wav = normalize(wav) * 0.95
        wav = torch.FloatTensor(wav).unsqueeze(0)
        wav = wav.to(torch.device('cuda'))
        vq_codes = self.vqvae.encode(wav) # 
        syn = self.vqvae(vq_codes)
        return fid, syn

    @torch.no_grad()
    def vq(self, wav_path):
        wav, sr = sf.read(wav_path)
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        fid = os.path.basename(wav_path)[:-4]
        wav = normalize(wav) * 0.95
        wav = torch.FloatTensor(wav).unsqueeze(0)
        wav = wav.to(torch.device('cuda'))
        vq_codes = self.vqvae.encode(wav)
        
        return fid, vq_codes
