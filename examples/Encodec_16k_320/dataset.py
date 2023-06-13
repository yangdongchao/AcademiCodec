from torch.utils.data import Dataset
import torchaudio
import random
import glob
import torch
import os
def get_dataset_filelist(a):
    # a is a one element list
    with open(a[0], 'r') as f:
        training_files = [l.strip() for l in f]
    return training_files
class NSynthDataset(Dataset):
    """Dataset to load NSynth data."""
    
    def __init__(self, audio_dir):
        super().__init__()
        self.filenames = []
        self.filenames.extend(glob.glob(audio_dir+"/*.wav"))
        print(len(self.filenames))
        #assert 1==2
        _, self.sr = torchaudio.load(self.filenames[0])
        #print(self.sr)
        self.max_len = 19200 # 24000
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        #print(self.filenames[index])
        ans = torch.zeros(1, self.max_len)
        audio = torchaudio.load(self.filenames[index])[0]
        audio = audio * 0.95
        # print('audio ', audio.shape)
        # assert 1==2
        if audio.shape[1] > self.max_len:
            st = random.randint(0, audio.shape[1]-self.max_len-1)
            ed = st + self.max_len
            return  audio[:,st:ed]
        else:
            ans[:,:audio.shape[1]] = audio
            return ans
