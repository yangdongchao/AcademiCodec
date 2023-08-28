import glob
import random

import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path


class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""

    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = []
        for sub_dir in audio_dir:
            print(sub_dir)
            self.filenames.extend(list(Path(sub_dir).glob("**/*.wav")))
            
        print(len(self.filenames))
        _, self.sr = torchaudio.load(self.filenames[0])
        self.max_len = 24000  # 24000

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1] - self.max_len - 1)
            ed = st + self.max_len
            return audio[:, st:ed]
        else:
            ans[:, :audio.shape[1]] = audio
            return ans
