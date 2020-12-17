#!/bin/bash

stage=1

train_set=$1
split_file=$2
dev_set=$3
testset_task1=$4
testset_task2=$5
# kaldi type data preparation
if [ $stage -le 1 ];then
	echo "local/prepare_all.sh"
	cd ./src
	echo $PWD
	python prepare_data.py --dataset_path $train_set --dest_path data/all_train || exit 1
	python cut_keyword_wav.py --split_file $split_file --wav_file data/all_train/wav.scp --save_dir data/split_wav/ || exit 1
	python combine_data.py --split_wav_dir data/split_wav/ --data_dir data/all_train --combine_data_dir data/PVTC	|| exit 1
	python prepare_task_data.py --dev_dataset $dev_set --dest_dir ../task/ || exit 1
	./prepare_testset_data.sh $testset_task1 ../testset_task1
	./prepare_testset_data.sh $testset_task2 ../testset_task2	
	cd ../
	echo "local/prepare_all.sh succeeded"
fi





