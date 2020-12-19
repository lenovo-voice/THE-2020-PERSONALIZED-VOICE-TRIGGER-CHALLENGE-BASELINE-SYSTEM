#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
import time, os, itertools, shutil, importlib ,tqdm
from DatasetLoader import loadWAV
from tuneThreshold import tuneThresholdfromScore
from SpeakerNet import SpeakerNet
from DatasetLoader import get_data_loader
from sklearn import metrics
from scipy import spatial
from matplotlib import pyplot
import matplotlib.pyplot as plt

def convert_alpha(string):
    tmp =""
    alpha=[]
    for i in string:
        if i==',':
            a_t = float(tmp)
            alpha.append(a_t)
            tmp=''
        else:
            tmp+=i
    a_t = float(tmp)
    alpha.append(a_t)
    return alpha


parser = argparse.ArgumentParser(description = "SpeakerNet");

parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');

## Data loader
parser.add_argument('--max_frames',     type=int,   default=300,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=400,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.01,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=1e-4,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=1,      help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=15,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');
parser.add_argument('--nClasses_ft',    type=int,   default=5994,   help='Number of speakers in the softmax layer, only for softmax-based losses');
## Load and save
parser.add_argument('--start_epoch',    type=int,   default=1,     help='Initial model weights');
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_interval',  type=int,   default=1,     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="./data/exp1", help='Path for model and logs');
parser.add_argument('--finetune_model', type=str,   default="",     help="Finetune initial model")
parser.add_argument('--trial_mode',     type=bool,   default=True,     help="Finetune initial model")

## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="",     help='Evaluation list');
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer ');

## inference parameters
parser.add_argument('--trials_list',       type=str,   default="",     help='trials file for pvtc');
parser.add_argument('--utt2wav',        type=str,   default="",     help='wav.scp flie for trails data')
parser.add_argument('--uttpath',           type=str,   default="",     help='flie path for trails data')
parser.add_argument('--utt2label',         type=str,   default="",     help='judgement result from kws system')
parser.add_argument('--eolembd_save',      type=str,   default="",     help='save path for enrollment embds')
# parser.add_argument('--utt2wav_kws',    type=str,   default="",     help='wav.scp file from kws system')
parser.add_argument('--parameter_savepath',type=str,default="",     help='threthold dic for sv system')
parser.add_argument('--save_dic',          type=bool,  default=True,   help='whether save embds for enrollment data')
parser.add_argument('--alpha',             type=convert_alpha,   default='1,1.5,2,3,5,10,15,19,20' ,    help='Alpha');


## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
parser.add_argument('--inference', dest='inference' ,action='store_true', help='Using for inference')
args = parser.parse_args();

## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))

def compute_final_score(labels, scores, alpha):
    count_pos = 0
    count_neg = 0
    miss = 0
    fa = 0
    for i in labels:
        if i=='positive':
            count_pos = count_pos+1
        else:
            count_neg = count_neg+1
    for i,score in enumerate(scores):
        if (score=='negative' )&( labels[i]=='positive'):
            miss = miss+1
        elif (score =='positive') & (labels[i]=='negative'):
            fa = fa+1
    miss_rate = miss / count_pos
    far = fa / count_neg
    return miss_rate+ alpha*far , far, miss_rate





## Initialise directories
if not(os.path.exists(args.save_path)):
    os.makedirs(args.save_path)

## Load models
s = SpeakerNet(**vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [100];

## Load model weights

s.loadParameters(args.initial_model);
it          = args.start_epoch;
print("Model %s loaded!"%args.initial_model);


 


if args.inference == True:
    s.eval()
    # utt2wav = {line.split()[0]: line.split()[1] for line in open(args.utt2wav)}
    utt2label = {line.split()[0]: line.split()[1] for line in open(args.utt2label)}
    eolembd_savepath = args.save_path+'/enrollment_data.npy'
    if(os.path.exists(eolembd_savepath)):
        enroll_dic = numpy.load(eolembd_savepath,allow_pickle=True).item()
    else:
        enroll_dic = s.enrollment_dic_kwsTrials(args.trials_list,args.uttpath,utt2label,eolembd_savepath,10,save_dic=args.save_dic)
    with open(args.trials_list) as f :
        lines = f.readlines()
    output_score = []

    final_labels = []
    eval_dic = {}


    parameter_dic = numpy.load(args.save_path+args.parameter_savepath,allow_pickle=True).item()
    threshold = parameter_dic['eer_threshold']
    if len(lines[0].strip().split()) == 5: # on dev set
        print('Deal with dev set')
        tsatrt = time.time()
        for line in tqdm.tqdm(lines):
            data = line.strip().split()
            final_labels.append(data[4])
            if utt2label[data[3]] == 'non-trigger':
                output_score.append('negative')
                continue
            else:
                with torch.no_grad():
                    uttid = data[3] +'.wav'
                    if uttid not in eval_dic:
                        inp = torch.FloatTensor(loadWAV(args.uttpath + uttid,0,True,10)).cuda()
                        eval_embd = s.__S__.forward(inp).cpu().numpy()
                        eval_dic[uttid] = eval_embd
                    else:
                        eval_embd = eval_dic[uttid] 
                eval_embd = numpy.squeeze(eval_embd)
                enroll_embd = (enroll_dic[data[0]] + enroll_dic[data[1]] +enroll_dic[data[2]]) /3
                enroll_embd = numpy.squeeze(enroll_embd)
                result = 1 - spatial.distance.cosine(eval_embd, enroll_embd)
                if result < threshold:
                    output_score.append('negative')
                else:
                    output_score.append('positive')
        tend = time.time()- tsatrt
        print('total time: %.2f' %(tend))

        scores =[]
        for al in args.alpha:
            S_kws , _, _ = compute_final_score(final_labels,output_score,al)
            print('Alpha: %d : S_kws: %.5f ' %(al,S_kws))
            scores.append(S_kws)
        plt.plot(args.alpha, scores,label='S_kws')
        plt.legend()  
        plt.xlabel('Alpha') 
        plt.ylabel("Final score")
        taskid = args.trials_list.split('/')[-2] 
        plt.savefig(f'../S_kws_{taskid}.jpg')
        with open(f'../Baseline_{taskid}_01.txt','w') as f:
            for i,line in enumerate(lines):
                data = line.strip().split()
                if output_score[i]=='positive':
                    f.write(data[0]+' '+data[1]+' '+data[2]+' '+data[3]+' 1\n')
                else:
                    f.write(data[0]+' '+data[1]+' '+data[2]+' '+data[3]+' 0\n')
        with open(f'../S_kws_{taskid}.txt','w') as f:
            for i,s in enumerate(scores):
                f.write('Alpha:%.2f S_kws:%.5f \n'%(args.alpha[i],s))

    elif len(lines[0].strip().split()) == 4: #eval set without label
        print('Deal with eval set')
        utt2wav = {line.split()[0]: line.split()[1] for line in open(args.utt2wav)}
        tsatrt = time.time()
        enroll_dic = {}
        for line in tqdm.tqdm(lines):
            data = line.strip().split()
            for enroll_utt in data[0],data[1],data[2]:
                if enroll_utt not in enroll_dic:
                    with torch.no_grad():
                        inp = torch.FloatTensor(loadWAV(utt2wav[enroll_utt],0,True,10)).cuda()
                        enroll_embd =  s.__S__.forward(inp).cpu().numpy()
                        enroll_dic[enroll_utt] = enroll_embd
            if utt2label[data[3]] == 'non-trigger':
                output_score.append('negative')
                continue
            else:
                with torch.no_grad():
                    uttid = data[3] +'.wav'
                    if uttid not in eval_dic:
                        inp = torch.FloatTensor(loadWAV(args.uttpath + uttid,0,True,10)).cuda()
                        eval_embd = s.__S__.forward(inp).cpu().numpy()
                        eval_dic[uttid] = eval_embd
                    else:
                        eval_embd = eval_dic[uttid] 
                eval_embd = numpy.squeeze(eval_embd)
                enroll_embd = (enroll_dic[data[0]] + enroll_dic[data[1]] +enroll_dic[data[2]]) /3
                enroll_embd = numpy.squeeze(enroll_embd)
                result = 1 - spatial.distance.cosine(eval_embd, enroll_embd)
                if result < threshold:
                    output_score.append('negative')
                else:
                    output_score.append('positive')
        tend = time.time()- tsatrt
        print('total time: %.2f' %(tend))
        taskid = args.trials_list.split('/')[-2] 
        with open(f'../Baseline_{taskid}_01.txt','w') as f:
            for i,line in enumerate(lines):
                data = line.strip().split()
                if output_score[i]=='positive':
                    f.write(data[0]+' '+data[1]+' '+data[2]+' '+data[3]+' 1\n')
                else:
                    f.write(data[0]+' '+data[1]+' '+data[2]+' '+data[3]+' 0\n')

    quit()






