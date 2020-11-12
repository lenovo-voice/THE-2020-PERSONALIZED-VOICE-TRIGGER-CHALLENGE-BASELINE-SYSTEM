from utils.file_tool import read_file_gen
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import soundfile as sf
import string
import random
import math
import sys
import os
import argparse


parser = argparse.ArgumentParser(description = "Cut keyword wavs and save them");
parser.add_argument('--split_file', type=str,default=None,dest='split_file',help='Official time index file');
parser.add_argument('--wav_file', type=str,default=None,dest='wav_file',help='Wav file');
parser.add_argument('--save_dir', type=str,default=None,dest='save_dir',help='Wavs destination path');
args = parser.parse_args();

ctm_files = [args.split_file] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/exp/tri4b_dnn_mpe/decode_test_word_it3/ctm"]
wavscps = [args.wav_file] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/data/test/wav.scp"]
save_dir = args.save_dir
beg_context = 0 # 3600
end_context = 0 # 1200

os.system("mkdir -p "+save_dir+" && mkdir -p "+save_dir+"/xiaole/" + " && mkdir -p "+save_dir+"/other/")

def read_utt2wav():
    utt2wav = {}
    for wavscp in wavscps:
        curr_utt2wav = dict({line.split()[0]:line.split()[1] for line in open(wavscp)})
        # merge dict
        utt2wav = {**utt2wav, **curr_utt2wav}
    print("utt2wav:", len(list(utt2wav)))
    return utt2wav

utt2wav = read_utt2wav()

def read_signal(utt_id):
    utt_file_path = utt2wav[utt_id]
    signal, sr = sf.read(utt_file_path)
    return signal, sr

def save_feat(feat, word, utt_id):
    word_save_dir = save_dir + '/' + word + '_'
    np.save(word_save_dir + utt_id, feat)

def sig_index_to_feat_index(sig_beg):
    fea_beg = max(0, math.floor(((sig_beg - 0.015) / 0.01) + 1))
    return fea_beg

def cut_word_and_save(items):
    utt_id = items[0]
    word = items[1]
    tbegin = items[2]
    tend = items[3]

    word_save_dir = save_dir + '/xiaole/' + word + '_'
    neg_word_save_dir = save_dir + '/other/other_'
    if os.path.exists(word_save_dir + utt_id + ".wav") and os.path.exists(neg_word_save_dir + utt_id + ".wav"):
        return 0
    sig, sr = read_signal(utt_id)
    #sig = sig[:,0]
    if len(sig.shape) > 1:
        keyword_sample = sig[int(sr*tbegin):int(sr*tend),:]
        neg_sample = sig[int(sr*tend):,:]
        sf.write(word_save_dir+ utt_id + ".wav",keyword_sample,sr)
        sf.write(neg_word_save_dir + utt_id + ".wav",neg_sample,sr)
    else:
        keyword_sample = sig[int(sr*tbegin):int(sr*tend)]
        neg_sample = sig[int(sr*tend):]
        sf.write(word_save_dir+ utt_id + ".wav",keyword_sample,sr)
        sf.write(neg_word_save_dir + utt_id + ".wav",neg_sample,sr)
    return 1

def get_words_list(ctm_file):
    word_segments = []
    for line in open(ctm_file):
        item = line.split()
        utt = '-'.join(item[0].split("/")[1:]).replace(".wav","")
        word_segments.append([utt,item[1],float(item[2]),float(item[3])])
    return word_segments

def extract_words(ctm_file):
    process_num = 30
    word_segments = get_words_list(ctm_file)
    print("word_segments:", len(word_segments))
    # print(word_segments)
    with mp.Pool(process_num) as p:
        frames = list(tqdm(p.imap(cut_word_and_save, word_segments), total=len(word_segments)))

    print(sum(frames), len(frames))
    print("Done.")

for ctm_file in ctm_files:
    extract_words(ctm_file)


