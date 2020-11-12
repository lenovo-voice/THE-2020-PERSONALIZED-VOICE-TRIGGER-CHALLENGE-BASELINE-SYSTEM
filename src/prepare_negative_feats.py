import os
import sys
from os import path
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import string
import random
from utils.feats import logFbankCal
import librosa
import torch
import soundfile as sf
import argparse

parser = argparse.ArgumentParser(description = "Negitive features preparation");
parser.add_argument('--wavfile_path', type=str,default=None,dest='wavfile_path',help='Wavs path path');
parser.add_argument('--dest_path',type=str,default=None,dest='dest_path',help='Save destination path');
args = parser.parse_args();

input_dir = args.wavfile_path
output_dir = args.dest_path
wav_list = [(line.strip().split()[0], line.strip().split()[1]) for line in open(input_dir)]
os.system("mkdir -p "+output_dir)

def extract_and_save(items):
    wav_id = items[0]
    wav_path = items[1]

    if path.exists(output_dir + "/" + wav_id + ".npy"): return 1
    try:
        wav, sr = sf.read(wav_path)
        #wav, _ = librosa.load(wav_path, 16000)
        if len(wav.shape) > 1:
            wav = wav[:, 0]
        featCal = logFbankCal(sample_rate = 16000,n_fft = 512,win_length=int(16000*0.025),hop_length=int(16000*0.01),n_mels=80)
        feats = featCal(torch.from_numpy(wav).float())
        np.save(output_dir + "/" + wav_id + ".npy", feats)
	check = np.load(output_dir + "/" + wav_id + ".npy")
    except:
        print(wav_id)
        return 2
 
    return 0

def extract_words(wav_scp):
    process_num = 20
    # print(word_segments)
    with mp.Pool(process_num) as p:
        _ = list(tqdm(p.imap(extract_and_save, wav_scp), total=len(wav_scp)))
    print("Done.")


extract_words(wav_list)
