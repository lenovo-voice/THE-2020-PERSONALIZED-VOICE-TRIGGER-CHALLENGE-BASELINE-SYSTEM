#!/bin/bash

stage=1

train_set=$1
split_file=$2
dev_set=$3

# kaldi type data preparation
if [ $stage -le 1 ];then
	cd ./src
	echo $PWD
	python prepare_data.py --dataset_path $train_set --dest_path data/all_train || exit 1
	python cut_keyword_wav.py --split_file $split_file --wav_file data/all_train/wav.scp --save_dir data/split_wav/ || exit 1
	python combine_data.py --split_wav_dir data/split_wav/ --data_dir data/all_train --combine_data_dir data/PVTC	|| exit 1
	python prepare_task_data.py --dev_dataset $dev_set --dest_dir ../task/ || exit 1
	cd ../
fi

# align words time index
if [ $stage -le 2 ];then
	cd ./src
	echo $PWD
	./align_nnet3_word.sh
	cd ../
fi

# prepare keywords train set
if [ $stage -le 3 ];then
	mkdir -p data
	cd ./src
	echo $PWD
	python prepare_keyword_feats.py --ctm_file exp/nnet3_PVTC/ctm --wavfile_path data/PVTC_nopitch/wav.scp --save_dir ../data/train_feat/positive
	cd ../
fi

# preapre negtive set
if [ $stage -le 4 ];then
	cd ./src
	python prepare_negative_feats.py --wavfile_path data/PVTC/neg_wav.scp --dest_path ../data/train_feat/negative || exit 1
	cd ../
fi















