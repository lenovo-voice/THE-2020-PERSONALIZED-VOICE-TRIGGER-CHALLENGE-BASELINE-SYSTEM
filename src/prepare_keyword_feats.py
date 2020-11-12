from utils.file_tool import read_file_gen
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
#import soundfile as sf
import librosa
import string
import random
import math
import sys
import os
from utils.feats import logFbankCal
import torch
import argparse

parser = argparse.ArgumentParser(description = "Keyword features preparation");
parser.add_argument('--ctm_file', type=str,default=None,dest='ctm_file',help='Align ctm file path');
parser.add_argument('--wavfile_path',type=str,default=None,dest='wavfile_path',help='Wavs path file');
parser.add_argument('--save_dir',type=str,default=None,dest='save_dir',help='Save destination path');
args = parser.parse_args();


ctm_files = [args.ctm_file] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/exp/tri4b_dnn_mpe/decode_test_word_it3/ctm"]
wavscps = [args.wavfile_path] # "/NASdata/zhangchx/kaldi/egs/thchs30/s5/data/test/wav.scp"]
save_dir = args.save_dir
beg_context = 0 # 3600
end_context = 0 # 1200

os.system("mkdir -p " + save_dir)

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
    signal,_ = librosa.load(utt_file_path, 16000)
    #signal, sr = sf.read(utt_file_path)
    return signal, 16000

def save_feat(feat, word, utt_id):
    word_save_dir = save_dir + '/' + word + '_'
    np.save(word_save_dir + utt_id, feat)

def sig_index_to_feat_index(sig_beg):
    fea_beg = max(0, math.floor(((sig_beg - 0.015) / 0.01) + 1))
    return fea_beg

def cut_word_and_save(items):
    utt_id = items[0]
    word = items[1]
    tmid = items[2]
    # print(items)

    word_save_dir = save_dir + '/' + word + '_'
    if os.path.exists(word_save_dir + utt_id + ".npy"):
        return 0

    sig, sr = read_signal(utt_id)
    if len(sig.shape) > 1:
        sig = sig[:, 0]
    nsig = sig
    featCal = logFbankCal(sample_rate = 16000,n_fft = 512,win_length=int(16000*0.025),hop_length=int(16000*0.01),n_mels=80)
    feats = featCal(torch.from_numpy(sig).float())
    feats = np.array(feats).T
    tmid = float(tmid)
    fea_mid = sig_index_to_feat_index(tmid)
    # while fea_beg + 40 < fea_end:
    #     cur_feat = feats[fea_beg:fea_beg+40]
    #     fea_beg += 1
    win = 20
    #print(1)
    while len(feats[fea_mid - win - 1: fea_mid + win]) < 40:
        win += 1
    # print(items, fea_mid - win, fea_mid + win)
    #print(3)
    feats = feats[fea_mid - win - 1 : fea_mid + win]
    feats = feats[0:40]
    feats = feats.T
    try:
        save_feat(feats, word, utt_id)
    except:
        print(utt_id)
        return 1
    return 1

def get_words_list(ctm_file):
    content_dict = {}
    word_segments = []
    print("get_words_list")
    
    for index, items in tqdm(read_file_gen(ctm_file)):
        if items[0] not in content_dict.keys():
           content_dict[items[0]] = {}
        #print(items)
        if items[4] in content_dict[items[0]].keys():
            content_dict[items[0]][items[4] + "#"] = items
        else:
            content_dict[items[0]][items[4]] = items

    for utt_id in content_dict.keys():
        content = content_dict[utt_id]
        try: 
            word_segments.append([utt_id, "xiaole", float(content["小"][2]) + float(content["小"][3])])
            word_segments.append([utt_id, "lexiao#", float(content["乐"][2]) + float(content["乐"][3]) ])
            word_segments.append([utt_id, "xiao#le#",  float(content["小#"][2]) + float(content["小#"][3])])
        except:
            print(utt_id)
    return word_segments

def extract_words(ctm_file):
    process_num = 20
    word_segments = get_words_list(ctm_file)
    print("word_segments:", len(word_segments))
    # print(word_segments)
    with mp.Pool(process_num) as p:
        frames = list(tqdm(p.imap(cut_word_and_save, word_segments), total=len(word_segments)))

    print(sum(frames), len(frames))
    print("Done.")

for ctm_file in ctm_files:
    extract_words(ctm_file)


