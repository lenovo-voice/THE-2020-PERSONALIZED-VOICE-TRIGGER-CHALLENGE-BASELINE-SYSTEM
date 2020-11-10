import os
import random
import numpy as np
import math
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset
from python_speech_features import mfcc, logfbank
from torch.utils.data import DataLoader
import time
class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, num_workers=0):
        self.bs = batch_size
        self.n_samples = len(dataset)
        self.num_workers = num_workers
        super(self.__class__, self).__init__(batch_sampler=self.batch_gen(), num_workers=self.num_workers, dataset=dataset)

    def __len__(self):
        return math.floor(self.n_samples / self.bs)

    def batch_gen(self):
        for i in range(math.floor(self.n_samples / self.bs)):
            ind = np.arange(self.bs).reshape(self.bs, 1) + i * self.bs
            yield ind

class NpyDataset(Dataset):
    def __init__(self, utt2data, utt2label=None, label2int=None, need_aug=False, with_label=True, shuffle=True, win_size=121):
        self.utt2data = utt2data
        self.dataset_size = len(self.utt2data)
        self.shuffle = shuffle
        self.with_label = with_label
        self.utt2label = utt2label
        self.label2int = label2int
        self.need_aug = need_aug
        self.win_size = win_size

        if self.with_label:
            assert self.utt2label and self.label2int is not None, "utt2label must be provided in with_label model! "

        if shuffle:
            random.shuffle(self.utt2data)

    def __len__(self):
        return self.dataset_size

    def read_feat(self, filename):

        samples = self.win_size
        try:
            feat = np.load(filename)
        except:
            print(filename)
        if feat.shape[1] == samples:
            new_feat = feat
        elif feat.shape[1] > samples:
            start_point = random.randrange(0, feat.shape[1] - samples)
            new_feat = np.array(feat[:,start_point:start_point + samples])
            #print(new_feat.shape)
        else:
            new_feat = np.zeros((80, samples))
            pad_beg = int((samples - feat.shape[1])/ 2)
            new_feat[:,pad_beg:pad_beg + feat.shape[1]] = feat
            #print(new_feat.shape)
        assert new_feat.shape == (80,self.win_size)
        return new_feat      

    def __getitem__(self, sample_idx):
        idx = int(sample_idx[0])
        #idx = sample_idx
        assert 0 <= idx and idx < self.dataset_size, "invalid index"
        
        utt, filename = self.utt2data[idx]
        feat = self.read_feat(filename)
        feat = feat.astype('float32')
        #print(feat.shape)
        if self.with_label:
            return utt, feat, int(self.label2int[self.utt2label[utt]])
        else:
            return utt, feat

def test():
    batch_size = 32
    num_workers = 6
    index_root_dir = './index_phone/'
    utt2wav = dict([line.split() for line in open(index_root_dir + 'train_utt2npy')])
    utt2label = dict([line.split() for line in open(index_root_dir + 'utt2label')])
    label2int = dict([line.split() for line in open(index_root_dir + 'label2int')])
    utt2data = []

    print(label2int)

    for k in utt2wav.keys():
        if utt2label[k] == 'orca':
            utt2data.append([k, utt2wav[k]])
            utt2data.append([k, utt2wav[k]])
        else:
            utt2data.append([k, utt2wav[k]])

   
    dev_dataset = NpyDataset(utt2data, utt2label, label2int, need_aug=True, with_label=True, shuffle=False, win_size=20)
    dear_dev_dataloader = MyDataLoader(dev_dataset, batch_size=batch_size, num_workers=1)

    print("Data loader prepared...")
    for utt, feat, l in dear_dev_dataloader:
        # print(utt)
        print(feat.shape, l)
        time.sleep(1)
if __name__ == "__main__":
    test()
