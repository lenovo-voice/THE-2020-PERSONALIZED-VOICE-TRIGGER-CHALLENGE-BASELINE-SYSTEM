import torch, torch.nn as nn, random
from torchaudio import transforms
import torch.nn.functional as F
import numpy as np
from librosa.filters import mel as librosa_mel_fn
import librosa
'''
TTS standard hyper-parameters for 16k audios
sample_rate = 16000
n_fft = 800
win_size = 800
hop_size = 200
mel_bins = 80
fmin = 55
fmax = 7600
rescale = True
rescaling_max = 0.9
max_abs_value = 4.
preemphasis = 0.97
preemphasize = True
min_level_db = -100
ref_level_db = 20
symmetric_mels = True
'''

class logFbankCal(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=800, win_length=800, hop_length=200, 
                 n_mels=80, rescale=True, rescaling_max=0.9, max_abs_value=4.,
                 preemphasis=0.97, preemphasize=True, fmin=55, fmax=7600,
                 min_level_db=-100, ref_level_db=20, symmetric_mels=True):
        super(logFbankCal, self).__init__()
        
        # these basic hyparams can be removed
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
        self.rescale = rescale
        self.rescaling_max = torch.tensor(rescaling_max, dtype=torch.float)
        self.preemphasize = preemphasize
        self.flipped_filter = torch.FloatTensor([-preemphasis, 1.]).unsqueeze(0).unsqueeze(0)
        
        self.stftCal = transforms.Spectrogram(n_fft, win_length, hop_length, power=None)
        mel_basis = librosa_mel_fn(sample_rate, n_fft, n_mels, fmin, fmax)
        self.mel_basis = torch.from_numpy(mel_basis).float()
        
        self.symmetric_mels = symmetric_mels
        self.ref_level_db = torch.tensor(ref_level_db, dtype=torch.float)
        self.min_level_db = torch.tensor(min_level_db, dtype=torch.float)
        self.min_level = torch.tensor(np.exp(min_level_db / 20 * np.log(10)), dtype=torch.float)
        self.max_abs_value = torch.tensor(max_abs_value, dtype=torch.float)
        
    def forward(self, x, is_aug=[]):
        # Signal Rescale
        if self.rescale:
            x = x / x.abs().max() * self.rescaling_max
        
        # Preemphasis
        #print("x.size()",x.size())
        if self.preemphasize:
            x = x.unsqueeze(0).unsqueeze(0)
            #print("x[:10]: ",x[:10])
            #print("unzqueeze x.size()",x.size())
            x = F.pad(x, (1, 0), 'reflect')
            #print("x[:10]: ",x[:,:,:10])
            x = F.conv1d(x, self.flipped_filter).squeeze()
            #print("preem x.size()",x.size())
            #print("x[:10]: ",x[:10])
        # STFT
        d = self.stftCal(x)
        #print("d.size()",d.size())
        #print("d.norm.size()",d.norm(p=2, dim=-1).size()) 
        # Filter banks
        #print("self.mel_basis.size()",self.mel_basis.size())
        _mel = self.mel_basis.matmul(d.norm(p=2, dim=-1))
        
        s = 20 * torch.log10(torch.max(self.min_level, _mel)) - self.ref_level_db
        
        # Symmetry and Clip
        if self.symmetric_mels:
            out = (2 * self.max_abs_value) * ((s - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
            out = torch.min(self.max_abs_value, torch.max(-self.max_abs_value, out))
        else:
            out = elf.max_abs_value * ((s - self.min_level_db) / (-self.min_level_db)) - self.max_abs_value
            out = torch.min(self.max_abs_value, torch.max(0, out))
        out = np.array(out)    
        return out


def get_feat(wav_path):
    wav, _ = librosa.load(wav_path, 16000)
    torch_fb = logFbankCal(16000, 800)
    torch_mel = torch_fb(torch.from_numpy(wav).float())
    return torch_mel

if __name__ == "__main__": 
# Example
    wavfile = '/mingback/wuhw/new_code/hotword_mia/data/data3_0/S0001_001I0.25M_1_0167_normal.wav'
    wav, _ = librosa.load(wavfile, 16000)
    torch_fb = logFbankCal(16000, 800)
    torch_mel = torch_fb(torch.from_numpy(wav).float())
    #output = np.array(torch_mel)
    print(torch_mel.shape)
    print(wav.shape[0]/16000)
    print((torch_mel.shape[1]*200+600)/16000)
    #print(output.shape)
