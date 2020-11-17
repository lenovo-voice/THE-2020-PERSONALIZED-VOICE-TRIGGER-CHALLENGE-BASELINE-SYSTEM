#!/usr/bin/python
#-*- coding: utf-8 -*-


import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile


## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description = "Train file generater");


parser.add_argument('--train_set', 	     type=str, default="data", help='PVTC train path')
parser.add_argument('--dev_path', 	     type=str, default="data", help='PVTC dev path')
parser.add_argument('--pvtc_trials_path',type=str, default="data", help='task1 trials')
parser.add_argument('--utt2label',       type=str, default="data", help='trials utt2label template')
parser.add_argument('--split_path',      type=str, default="data", help='split data path for dev')

parser.add_argument('--make_sv_trials',  dest='make_sv_trials',  action='store_true', help='Make sv trials')
parser.add_argument('--make_list', dest='make_list', action='store_true', help='Make finetune train list')



args = parser.parse_args();


def generate_list(args):
	files_pure = glob.glob('../src/data/split_wav/xiaole/*.wav')
	files_all = glob.glob('%s/*/*/*/*.wav'%args.train_set)
	list_pure = []
	list_all = []
	for line in (files_pure):
		list_pure.append(line.split('/')[5].split('_')[1].split('-')[0]+' '+line)
	for line in (files_all):
		list_all.append(line.split('/')[-4]+' '+line)

	return list_pure,list_all
	

def get_sv_trials(args):
	utt2label = {l.split()[0]:l.split()[1] for l in open(args.utt2label)}
	with open(args.pvtc_trials_path) as f:
		lines = f.readlines()
	sv_trial = []
	for line in lines:
		test_data = line.split()[3]
		if utt2label[test_data] == 'positive':
			if line.split()[4] == 'positive':
				sv_trial.append('1 '+args.dev_path+line.split()[0]+' '+args.split_path+test_data)
			else:
				sv_trial.append('0 '+args.dev_path+line.split()[0]+' '+args.split_path+test_data)
	return sv_trial

## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":

	if not os.path.exists(args.train_set):
		raise ValueError('Target directory does not exist.')

		
	if args.make_list:
		print('make train lists')
		list_pure,list_all = generate_list(args)
		with open('list_pvtc_all','w') as f:
			for line in list_all:
				f.write(line.strip()+'\n')
		with open('list_pvtc_pure','w') as f:
			for line in list_pure:
				f.write(line.strip()+'\n')

	if args.make_sv_trials:
		print('make trials')
		sv_trials = get_sv_trials(args)
		with open('sv_trials','w') as f:
			for line in sv_trials:
				f.write(line.strip()+'\n')
		