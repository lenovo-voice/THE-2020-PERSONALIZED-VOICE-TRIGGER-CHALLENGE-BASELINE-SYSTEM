import pickle
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import roc_curve, auc
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
parser.add_argument('--model_class',type=str,default='lstm_models',dest='model_class',help='Model class');
parser.add_argument('--model_name',type=str,default='LSTMAvg',dest='model_name',help='Model name');
parser.add_argument('--word_num',type=int,default=3,dest='word_num',help='Decoding word num');
parser.add_argument('--step_size',type=int,default=3,dest='step_size',help='Decoding step size');
parser.add_argument('--conf_size',type=int,default=150,dest='conf_size',help='Decoding confidence size');
parser.add_argument('--vad_mode',type=int,default=3,dest='vad_mode',help='Vad mode');
parser.add_argument('--vad_max_length',type=int,default=130,dest='vad_max_length',help='Vad max length');
parser.add_argument('--vad_max_activate',type=float,default=0.9,dest='vad_max_activate',help='Vad max activate');
parser.add_argument('--txt_name',type=str,default='outputs_txts/Baseline-words_fbank8040_LSTMAvg_test_task1.txt',dest='txt_name',help='Txt file name')
parser.add_argument('--save_path',type=str,default='data/trigger_wav/test/task1/',dest='save_path',help='Saving wav path')
parser.add_argument('--threshold',type=float,default=0.0413,dest='th',help='Threshold')
parser.add_argument('--predict_length',type=int,default=80,dest='predict_length',help='Vad predict length');
parser.add_argument('--segment_step',type=int,default=10,dest='segment_step',help='Vad segment step');
parser.add_argument('--wav_scp_file',type=str,default='',dest='wav_scp_file',help='task1 or task2')



args = parser.parse_args();

model_name = args.test_model
# test data
test_utt2wav=dict({line.strip().split()[0]:line.strip().split()[1] for line in open(args.wav_scp_file,encoding="utf-8")})
# models
# parameter
word_num = args.word_num
win_size= 40
step_size= args.step_size
smooth_size= 50
conf_size= args.conf_size
result = args.txt_name # Baseline-words_fbank8040_LSTMAvg_task1.txt
save_path = args.save_path
th = args.th 
segment_step = args.segment_step
predict_length = args.predict_length
os.system("mkdir -p {}".format(save_path))

f = open(result,"w")
os.system("mkdir -p outputs_pkls")

# get model
net = importlib.import_module('models.'+args.model_class).__getattribute__(args.model_name)(output_dim=word_num+1)
net = returnCPUModel(net, model_name)
net.eval()
print(net)
print("Model done.")

total_time = 0
times = []

for index,wav_key in enumerate(tqdm(test_utt2wav.keys())):

    # feature extraction
    wav_path = test_utt2wav[wav_key]
    confidence_list = []
    signal_list = []
    if ".wav" in wav_path:
        starttime = time.time()
        sum_confidence = []
        audio, sample_rate = read_wave(wav_path)
        all_signal = raw_to_float(audio)
        vad = webrtcvad.Vad(args.vad_mode)
        frames = frame_generator(30, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 30, args.vad_max_length, vad, frames, args.vad_max_activate)
        for seg_i in segments:
            seg_confidence = []
            signal = raw_to_float(seg_i)
            signal_list.append(seg_i)
            torch_fb = logFbankCal(sample_rate = 16000,n_fft = 512,win_length=int(16000*0.025),hop_length=int(16000*0.01),n_mels=80)
            raw_feats = torch_fb(torch.from_numpy(signal).float(),sub_mean=False)
            raw_feats = np.array(raw_feats)
            for seg in range(0, raw_feats.shape[1], segment_step):
                feats = []
                if seg + segment_step <= raw_feats.shape[1]:
                    temp_feats = raw_feats[:,seg:seg+predict_length]
                else:
                    temp_feats = raw_feats[:,raw_feats.shape[1]-predict_length:raw_feats.shape[1]]
                    break
                mean_feat = np.expand_dims(temp_feats.mean(axis=1), axis=1)
                temp_feats = temp_feats - mean_feat
                # sliding window
                feats = []
                for i in range(0, temp_feats.shape[1], step_size):
                    if i + win_size <= temp_feats.shape[1]:
                        cur_feat = temp_feats[:,i:i+win_size]
                    else:
                        cur_feat = temp_feats[:,temp_feats.shape[1]-win_size:temp_feats.shape[1]]
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
                seg_confidence.append(confidence)
            try:
                seg_max_conf = max(seg_confidence)
            except:
                seg_max_conf = 0.0
            sum_confidence.append(seg_max_conf)
    try:
        confidence = max(sum_confidence)
    except:
        confidence = 0
    if confidence >= th:
        max_index = sum_confidence.index(confidence)
        write_wave(save_path+"/"+wav_key+".wav",signal_list[max_index],16000)
        f.writelines("{} trigger\n".format(wav_key))
    else:
        f.writelines("{} non-trigger\n".format(wav_key))
    endtime = time.time()
    times.append(round(endtime - starttime, 2))
    if index % 100 == 0:
        tqdm.write("total processing time:"+str(sum(times))+"sec")
        tqdm.write(wav_key+ " " +str(confidence))
print("total processing time:",sum(times))
print("Predict done.")
print("Finished.")

