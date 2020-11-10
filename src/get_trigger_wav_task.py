import os
import sys
from tqdm import tqdm
from utils.vad_ext import *
import webrtcvad
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
import argparse
import importlib
parser = argparse.ArgumentParser(description = "SpeakerNet");
parser.add_argument('--test_model',type=str,default='outputs/train_Baseline-words_fbank8040_LSTMAvg/models/model_100',dest='test_model',help='Model path for testing');
parser.add_argument('--txt_name',type=str,default='outputs_txts/Baseline-words_fbank8040_LSTMAvg_task1.txt',dest='txt_name',help='Txt file name')
parser.add_argument('--mode',type=str,default='task1',dest='mode',help='task1 or task2')
parser.add_argument('--save_path',type=str,default='',dest='save_path',help='Saving wav path')
parser.add_argument('--threshold',type=float,default=None,dest='th',help='Threshold')
parser.add_argument('--model_class',type=str,default='lstm_models',dest='model_class',help='Model class');
parser.add_argument('--model_name',type=str,default='LSTMAvg',dest='model_name',help='Model name');
parser.add_argument('--word_num',type=int,default=3,dest='word_num',help='Decoding word num');
parser.add_argument('--step_size',type=int,default=3,dest='step_size',help='Decoding ste size');
parser.add_argument('--conf_size',type=int,default=150,dest='conf_size',help='Decoding confidence size');
parser.add_argument('--vad_mode',type=int,default=3,dest='vad_mode',help='Vad mode');
parser.add_argument('--vad_max_length',type=int,default=130,dest='vad_max_length',help='Vad max length');
parser.add_argument('--vad_max_activate',type=float,default=0.9,dest='vad_max_activate',help='Vad max activate');
args = parser.parse_args();



model = args.test_model #LSTM
mode = args.mode #task1 task2
result = args.txt_name # Baseline-words_fbank8040_LSTMAvg_task1.txt
save_path = args.save_path
th = args.th 

os.system("mkdir -p {}".format(save_path))

result = [item.split() for item in open(result)]
#result = [["filler_xiaole-PART1-0331-2-0448_fllowed_with_xiaole-PART1-0331-2-0448","trigger",0.19]]
utt2wav = dict(item.split() for item in open("./task/{}/utt2wav".format(args.mode)))
word_num = args.word_num
#pkl_name = "outputs_pkls/train_baseline_realdata_fbank80121_{0}_test_realdata_mixsyn.pkl".format(model)
win_size= 40
step_size= args.step_size
smooth_size= 50
conf_size= args.conf_size

net = importlib.import_module('models.'+args.model_class).__getattribute__(args.model_name)(output_dim=word_num+1)
net = returnCPUModel(net, model)
net.eval()

for item in tqdm(result):
    if item[1] == "trigger":
        confidence_list = []
        wav_path = utt2wav[item[0]]
        audio, sample_rate = read_wave(wav_path)
        vad = webrtcvad.Vad(args.vad_mode)
        frames = frame_generator(30, audio, sample_rate)
        segments = vad_collector(sample_rate, 30, args.vad_max_length, vad, frames,args.vad_max_activate)
        signal_list = []
        for seg in segments:
            signal = raw_to_float(seg)
            signal_list.append(seg)
            torch_fb = logFbankCal(sample_rate = 16000,n_fft = 512,win_length=int(16000*0.025),hop_length=int(16000*0.01),n_mels=80)
            raw_feats = torch_fb(torch.from_numpy(signal).float())
            raw_feats = np.array(raw_feats)
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
                confidence_list.append(0.0)
                continue
            out = predict(net, feats)

            smooth_out = out
            if word_num != 1:
                confidence = get_full_conf(smooth_out, conf_size, word_num)
            else:
                confidence = out[:,1].max()
            confidence_list.append(confidence)
        confidence = max(confidence_list)
        assert confidence >= th
        index = confidence_list.index(confidence)
        tqdm.write(item[0]+".wav"+ " " +str(confidence))
        write_wave(save_path+"/"+item[0]+".wav",signal_list[index],16000)




