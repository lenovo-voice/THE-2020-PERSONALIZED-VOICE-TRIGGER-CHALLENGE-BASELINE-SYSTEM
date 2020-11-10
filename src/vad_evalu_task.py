# -*- coding: utf-8 -*-
import pickle
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import roc_curve, auc
# from utils.plot_det import plot_det
from utils.decode_utils import predict, smooth, get_full_conf, get_full_conf2
from utils.model_utils import returnGPUModel, returnCPUModel
import soundfile as sf
from utils.save_utils import save_pickle
import os
import sys
import torch.nn as nn
import librosa
import soundfile as sf
from utils.feats import logFbankCal
import time
import random
from utils.vad_ext import *
import webrtcvad
import argparse
import importlib

parser = argparse.ArgumentParser(description = "SpeakerNet");
parser.add_argument('--test_model',type=str,default='outputs/train_Baseline-words_fbank8040_LSTMAvg/models/model_100',dest='test_model',help='Model path for testing');
parser.add_argument('--pickle_name',type=str,default='outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl',dest='pkl_name',help='Saving pickle file name')
parser.add_argument('--mode',type=str,default='task1',dest='mode',help='task1 or task2')
parser.add_argument('--model_class',type=str,default='lstm_models',dest='model_class',help='Model class');
parser.add_argument('--model_name',type=str,default='LSTMAvg',dest='model_name',help='Model name');
parser.add_argument('--word_num',type=int,default=3,dest='word_num',help='Decoding word num');
parser.add_argument('--step_size',type=int,default=3,dest='step_size',help='Decoding step size');
parser.add_argument('--conf_size',type=int,default=150,dest='conf_size',help='Decoding confidence size');
parser.add_argument('--vad_mode',type=int,default=3,dest='vad_mode',help='Vad mode');
parser.add_argument('--vad_max_length',type=int,default=130,dest='vad_max_length',help='Vad max length');
parser.add_argument('--vad_max_activate',type=float,default=0.9,dest='vad_max_activate',help='Vad max activate');
args = parser.parse_args();

model_name = args.test_model
pkl_name = args.pkl_name
mode = args.mode
# test data
test_utt2wav=dict({line.strip().split()[0]:line.strip().split()[1] for line in open("./{}/utt2wav".format(mode),encoding="utf-8")})
test_utt2label=dict({line.strip().split()[0]:line.strip().split()[1] for line in open("./{}/utt2label".format(mode),encoding="utf-8")})
test_label2int = dict({line.strip().split()[0]:line.strip().split()[1] for line in open("./{}/label2int".format(mode),encoding="utf-8")})
# models
# parameter
word_num = args.word_num
win_size= 40
step_size= args.step_size
smooth_size= 50
conf_size= args.conf_size

os.system("mkdir -p outputs_pkls")

# get model
#net = LSTMAvg(output_dim=4)
net = net = importlib.import_module('models.'+args.model_class).__getattribute__(args.model_name)(output_dim=word_num+1)
net = returnCPUModel(net, model_name)
net.eval()
print(net)
print("Model done.")

total_time = 0
y = []
scores = []
keys = []
preds = []
times = []
for index,wav_key in enumerate(tqdm(test_utt2wav.keys())):

    # feature extraction
    wav_path = test_utt2wav[wav_key]
    confidence_list = []
    if ".wav" in wav_path:
        starttime = time.time()
        
        audio, sample_rate = read_wave(wav_path)
        all_signal = raw_to_float(audio)
        vad = webrtcvad.Vad(args.vad_mode)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, args.vad_max_length, vad, frames, args.vad_max_activate)
        for seg in segments:
            signal = raw_to_float(seg)
            torch_fb = logFbankCal(sample_rate = 16000,n_fft = 512,win_length=int(16000*0.025),hop_length=int(16000*0.01),n_mels=80)
            raw_feats = torch_fb(torch.from_numpy(signal).float())
            raw_feats = np.array(raw_feats)

            # sliding window
            feats = []
            for i in range(0, raw_feats.shape[1], step_size):
                if i + win_size <= raw_feats.shape[1]:
                   cur_feat = raw_feats[:,i:i+win_size]
                else:
                   cur_feat = raw_feats[:,raw_feats.shape[1]-win_size:raw_feats.shape[1]]
                   feats.append(cur_feat)
                   break
                feats.append(cur_feat)
            if len(feats) <= word_num:
                continue
            out = predict(net, feats)

            smooth_out = out # [:-1,:] # smooth(out, smooth_size)
            if word_num != 1:
                confidence = get_full_conf(smooth_out, conf_size, word_num)
            else:
                confidence = out[:,1].max()
            confidence_list.append(confidence)
    try:
        confidence = max(confidence_list)
    except:
        confidence = 0
    pred = 1 if confidence > 0.0133 else 0
    endtime = time.time()
    times.append(round(endtime - starttime, 2))
    y.append(int(test_label2int[test_utt2label[wav_key]]))
    scores.append(confidence)
    preds.append(pred)
    keys.append(wav_key)
    if len(y) == 1: continue
    if index % 100 == 0:
        tqdm.write("total processing time:"+str(sum(times))+"sec")
        correct = sum(np.array(y) == np.array(preds))
        fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
        auc_score = auc(fpr, tpr)
        tqdm.write(wav_key + ":" + str(confidence) + " auc_score: " + str(auc_score) + " acc:" + str(correct / len(y)))
print("total processing time:",sum(times))
print("Predict done.")
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
auc_score = auc(fpr, tpr)
print("auc_score: ", auc_score)

save_pickle(pkl_name, {"y": y, "scores": scores, "keys": keys ,"time":times})
print("Finished.")




