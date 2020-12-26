import torch, torch.nn as nn, random
from torchaudio import transforms
import numpy as np
import librosa


class logFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(logFbankCal, self).__init__()
        self.fbankCal = transforms.MelSpectrogram(sample_rate=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels)

    def forward(self, x, is_aug=[],sub_mean=True):
        out = self.fbankCal(x)
        #print(out.size())
        out = torch.log(out + 1e-6)
        if sub_mean:
            out = out - out.mean(axis=1).unsqueeze(dim=1)
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
                #dim, alpha = 1, random.random() * 0.4 + 0.1
                #mask_dim = list(out[i].shape)
                #mask_dim[dim] = 1
                #mask_ots = [1 for i in mask_dim]
                #mask_ots[dim] = out[i].shape[dim]
                #out[i] = out[i] * torch.tensor(0.).repeat(mask_dim).uniform_(1-2*alpha, 1+2*alpha).repeat(mask_ots).cuda()
                
        return out


class FFToutCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(FFToutCal, self).__init__()
        self.complexSpec = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, 
                                                    hop_length=hop_length, power=None)
    
    def forward(self, x, is_aug=[]):
        out = self.complexSpec(x)
        specreal = out[...,0]
        specimag = out[...,1]
        out = torch.cat((specreal,specimag),dim=1)
        #out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return out

class FFToutSpliteCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(FFToutSpliteCal, self).__init__()
        self.complexSpec = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, 
                                                    hop_length=hop_length, power=None)
    
    def forward(self, x, is_aug=[]):
        out = self.complexSpec(x)
        specreal = out[...,0]
        specimag = out[...,1]
        #out = torch.cat((specreal,specimag),dim=1)
        #out = torch.log(out + 1e-6)
        specreal = specreal - specreal.mean(axis=2).unsqueeze(dim=2)
        specimag = specimag - specimag.mean(axis=2).unsqueeze(dim=2)
        
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return specreal ,specimag


class PhaseFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(PhaseFbankCal, self).__init__()
        self.complexSpec = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, 
                                                    hop_length=hop_length, power=None)
        self.mel_scale = transforms.MelScale(n_mels=n_mels ,sample_rate=sample_rate,n_stft= n_fft//2+1)
    
    def forward(self, x, is_aug=[]):
        out = self.complexSpec(x)
        real = out[...,0]
        imag = out[...,1]

        spec = torch.sqrt(real.pow(2)+imag.pow(2))
        phase = torch.atan2(imag,real)
        mel_spec = self.mel_scale(spec)
        mel_phase = self.mel_scale(phase)

        #out = torch.cat((specreal,specimag),dim=1)
        #out = torch.log(out + 1e-6)
        mel_spec = mel_spec - mel_spec.mean(axis=2).unsqueeze(dim=2)
        mel_phase = mel_phase - mel_phase.mean(axis=2).unsqueeze(dim=2)
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return torch.cat((mel_spec,mel_phase),dim=1)


class LogPhaseFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(LogPhaseFbankCal, self).__init__()
        self.complexSpec = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, 
                                                    hop_length=hop_length, power=None)
        self.mel_scale = transforms.MelScale(n_mels=n_mels ,sample_rate=sample_rate,n_stft= n_fft//2+1)
    
    def forward(self, x, is_aug=[]):
        out = self.complexSpec(x)
        real = out[...,0]
        imag = out[...,1]

        spec = (real.pow(2)+imag.pow(2))
        phase = torch.atan2(imag,real)+3.1415927
        mel_spec = self.mel_scale(spec)
        mel_phase = self.mel_scale(phase)

        out = torch.cat((mel_spec,mel_phase),dim=1)

        out = torch.log(out + 1e-6)
        out = out - out.mean(axis=2).unsqueeze(dim=2)
        #mel_spec = mel_spec - mel_spec.mean(axis=2).unsqueeze(dim=2)
        #mel_phase = mel_phase - mel_phase.mean(axis=2).unsqueeze(dim=2)
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return out


class LogSplitPhaseFbankCal(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels):
        super(LogSplitPhaseFbankCal, self).__init__()
        self.complexSpec = transforms.Spectrogram(n_fft=n_fft, win_length=win_length, 
                                                    hop_length=hop_length, power=None)
        self.mel_scale = transforms.MelScale(n_mels=n_mels ,sample_rate=sample_rate,n_stft= n_fft//2+1)
    
    def forward(self, x, is_aug=[]):
        out = self.complexSpec(x)
        real = out[...,0]
        imag = out[...,1]

        spec = (real.pow(2)+imag.pow(2))
        phase = torch.atan2(imag,real)+3.1415927 #change the angle into(0,2pi)
        mel_spec = self.mel_scale(spec)
        mel_phase = self.mel_scale(phase)

        mel_spec = torch.log(mel_spec + 1e-6)
        mel_phase = torch.log(mel_phase + 1e-6)
        #out = out - out.mean(axis=2).unsqueeze(dim=2)
        mel_spec = mel_spec - mel_spec.mean(axis=2).unsqueeze(dim=2)
        mel_phase = mel_phase - mel_phase.mean(axis=2).unsqueeze(dim=2)
        for i in range(len(is_aug)):
            if is_aug[i]:
                offset = random.randrange(out.shape[1]/8, out.shape[1]/4)
                start = random.randrange(0, out.shape[1] - offset)
                out[i][start : start+offset] = out[i][start : start+offset]  * random.random() / 2
        return mel_spec , mel_phase

if __name__ == "__main__":
    #wavfile = "/Netdata/AudioData/PVTC/process/vad_wav/小乐小乐_xiaole-PART1-0262-1-0010.wav"
    #wavfile = "/Netdata/AudioData/PVTC/process/vad_wav/小乐小乐_xiaole-PART1-0262-1-0345.wav"
    wavfile = "/Netdata/AudioData/PVTC/process/vad_wav/小乐小乐_xiaole-PART1-0262-4-0119.wav"
    wav, _ = librosa.load(wavfile, 16000)
    torch_fb = logFbankCal(sample_rate = 16000,n_fft = 512,win_length=int(16000*0.025),hop_length=int(16000*0.01),n_mels=80)
    feats = torch_fb(torch.from_numpy(wav).float())
    print(feats.size())
