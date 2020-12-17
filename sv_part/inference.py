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
from sklearn import metrics
from scipy import spatial


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
parser.add_argument('--trials_list',    type=str,   default="",     help='trials file for pvtc');
# parser.add_argument('--utt2wav',        type=str,   default="",     help='flie path for trails data')
parser.add_argument('--uttpath',        type=str,   default="",     help='flie path for trials split data')
parser.add_argument('--devdatapath',    type=str,   default="",     help='flie path for trials raw data')
parser.add_argument('--u2l_template',   type=str,   default="",     help='trials result from kws system')
parser.add_argument('--utt2label',      type=str,   default="",     help='judgement result from kws system')
parser.add_argument('--eolembd_save',   type=str,   default="",     help='save path for enrollment embds')
parser.add_argument('--save_dic',       type=bool,  default=True,    help='whether save embds for enrollment data')

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
    return miss_rate+ alpha*far

def get_final_score(threshold,scores,output_score):
    final_score = []
    count = 0 
    for i in output_score:
        if(i == 'tbd'):
            score = scores[count]
            if score >= threshold:
                final_score.append('positive')
            else: final_score.append('negative')
            count = count+1
        else:
            final_score.append(i)
    return final_score

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = numpy.concatenate((target_scores, nontarget_scores))
    labels = numpy.concatenate((numpy.ones(target_scores.size), numpy.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = numpy.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = numpy.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (numpy.arange(1, n_scores + 1) - tar_trial_sums)

    frr = numpy.concatenate((numpy.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = numpy.concatenate((numpy.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = numpy.concatenate((numpy.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
    return frr, far, thresholds

def ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold




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
    with open(args.trials_list) as f :
        lines = f.readlines()
    # utt2wav = {line.split()[0]: line.split()[1] for line in open(args.utt2wav)}
    utt2label = {line.split()[0]: line.split()[1] for line in open(args.utt2label)}
    u2l_template = {line.split()[0]: line.split()[1] for line in open(args.u2l_template)}
    eolembd_savepath = args.save_path+'/enrollment_data.npy'
    if(os.path.exists(eolembd_savepath)):
        enroll_dic = numpy.load(eolembd_savepath,allow_pickle=True).item()
    else:
        enroll_dic = s.enrollment_dic_kwsTrials(args.trials_list,args.devdatapath,utt2label,eolembd_savepath,10,save_dic=args.save_dic)

    output_score = []
    labels = []
    scores = []
    final_labels = []
    eval_dic = {}


    tsatrt = time.time()
    for line in tqdm.tqdm(lines):
        # Only use 40000 lines of trial file, because the back contains stitched audio
        data = line.strip().split()
        final_labels.append(data[4])
        if utt2label[data[3]] == 'negative':
            output_score.append('negative')
            continue
        elif (utt2label[data[3]] == 'trigger') & (u2l_template[data[3]] == 'positive'):
            with torch.no_grad():
                uttid = data[3]+'.wav'
                if uttid not in eval_dic:
                    inp = torch.FloatTensor(loadWAV(args.uttpath+uttid,0,True,10)).cuda()
                    eval_embd = s.__S__.forward(inp).cpu().numpy()
                    eval_dic[uttid] = eval_embd
                else:
                    eval_embd = eval_dic[uttid] 
            eval_embd = numpy.squeeze(eval_embd)
            enroll_embd = (enroll_dic[data[0]] + enroll_dic[data[1]] +enroll_dic[data[2]]) /3
            enroll_embd = numpy.squeeze(enroll_embd)
            result = 1 - spatial.distance.cosine(eval_embd, enroll_embd)
            scores.append(result)
            if data[4]  == 'negative':
                labels.append(0)
            else:
                labels.append(1)
            output_score.append('tbd')
        else:
            output_score.append('tbd')
    tend = time.time()- tsatrt
    print('total time: %.2f' %(tend))
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    fnr = fnr*100
    fpr = fpr*100
    auc = metrics.auc(fpr, tpr)
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])
    mindcf , min_c_det_threshold = ComputeMinDcf(fnr,fpr,thresholds)
    print('EER: %.4F ,Threshold： %.10F' %(eer,thresholds[idxE]))
    print('minDCF: %.4f ,Threshold： %.10F' %(mindcf*0.01 ,min_c_det_threshold))
    parameter_dic = {}
    # finalscore_dic = {}
    # parameter_dic['thresholds'] = thresholds
    print('Save threshold(EER threshold+mincDCF threshold /2)： %.10F' %((thresholds[idxE] +min_c_det_threshold) /2))
    parameter_dic['eer_threshold'] = (thresholds[idxE] +min_c_det_threshold) /2

    numpy.save(args.save_path+'/eer_threshold.npy',parameter_dic)


    quit()






