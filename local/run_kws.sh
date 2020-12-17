#!/bin/bash

stage=1

echo "local/run_kws.sh"

if [ $whether_train ];then
# align words time index
if [ $stage -le 1 ];then
	cd ./src
	echo $PWD
	./align_nnet3_word.sh || exit 1
	cd ../
fi

# prepare keywords train set
if [ $stage -le 2 ];then
	mkdir -p data
	cd ./src
	echo $PWD
	python prepare_keyword_feats.py --ctm_file exp/nnet3_PVTC/ctm --wavfile_path data/PVTC_nopitch/wav.scp --save_dir ../data/train_feat/positive || exit 1
	cd ../
fi

# preapre negtive set
if [ $stage -le 3 ];then
	cd ./src
	python prepare_negative_feats.py --wavfile_path data/PVTC/neg_wav.scp --dest_path ../data/train_feat/negative || exit 1
	cd ../
fi


if [ $stage -le 4 ];then
	python src/prepare_index.py --pos_feat_dir data/train_feat/positive --neg_feat_dir data/train_feat/negative --dest_dir index_words || exit 1
fi

if [ $stage -le 5 ];then
	python src/train_words_baseline.py --seed 40 --mode train --task_name Baseline-words --model_class lstm_models --model_name LSTMAvg --index_dir index_words --batch_size 128 --num_epoch 100 --lr 0.01 || exit 1
fi

output_model=outputs/train_Baseline-words_fbank8040_LSTMAvg/models/model_100

else

output_model=src/trained_kws_model

fi

if [ $stage -le 6 ];then
	python src/vad_evalu_task.py --test_model $output_model  --pickle_name outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl --mode task/task1 --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 3 --vad_max_length 130 --vad_max_activate 0.9 || exit 1
	python src/vad_evalu_task.py  --test_model $output_model --pickle_name outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task2.pkl --mode task/task2 --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 0 --vad_max_length 200 --vad_max_activate 0.8 || exit 1
fi

if [ $stage -le 7 ];then	
	mkdir -p outputs_txts
	python src/process_pkl_th.py --pickle_name outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl --txt_name outputs_txts/Baseline-words_fbank8040_LSTMAvg_task1.txt --threshold `python src/get_th.py --total_hours 20.1309 --plt_name wake_task1.jpg --pkl_names outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl --threshold_for_num_false_alarm_per_hour 1.0` || exit 1
	python src/process_pkl_th.py --pickle_name outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task2.pkl --txt_name outputs_txts/Baseline-words_fbank8040_LSTMAvg_task2.txt --threshold `python src/get_th.py --total_hours 36.0836 --plt_name  wake_task2.jpg --pkl_names outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task2.pkl --threshold_for_num_false_alarm_per_hour 1.0` || exit 1
fi

if [ $stage -le 8 ];then
	python src/get_trigger_wav_task.py --test_model $output_model --mode task1 --txt_name outputs_txts/Baseline-words_fbank8040_LSTMAvg_task1.txt --save_path data/trigger_wav/task1/ --threshold `python src/get_th.py --total_hours 20.1309 --plt_name wake_task1.jpg --pkl_names outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl --threshold_for_num_false_alarm_per_hour 1.0` --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 3 --vad_max_length 130 --vad_max_activate 0.9 || exit 1
	python src/get_trigger_wav_task.py  --test_model $output_model --mode task2 --txt_name outputs_txts/Baseline-words_fbank8040_LSTMAvg_task2.txt --save_path data/trigger_wav/task2/ --threshold `python src/get_th.py --total_hours 36.0836 --plt_name  wake_task2.jpg --pkl_names outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task2.pkl --threshold_for_num_false_alarm_per_hour 1.0` --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 0 --vad_max_length 200 --vad_max_activate 0.8 || exit 1

fi

if [ $stage -le 9 ];then
	python src/get_pos_wav_task.py --test_model  $output_model  --mode task/task1 --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 3 --vad_max_length 130 --vad_max_activate 0.9 --save_path data/trigger_wav/task1/
	python src/get_pos_wav_task.py --test_model  $output_model  --mode task/task2 --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 0 --vad_max_length 200 --vad_max_activate 0.8 --save_path data/trigger_wav/task2/

fi

if [ $stage -le 10 ];then
	
	python src/get_testset_trigger_wav_task.py --txt_name outputs_txts/Baseline-words_fbank8040_LSTMAvg_test_task1.txt --save_path data/trigger_wav/test/task1/ --wav_scp_file testset_task1/wav_for_wake.scp  --threshold `python src/get_th.py --total_hours 20.1309 --plt_name wake_task1.jpg --pkl_names outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl --threshold_for_num_false_alarm_per_hour 1.0` --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 3 --vad_max_length 130 --vad_max_activate 0.9 --predict_length 80 --segment_step 10

	python src/get_testset_trigger_wav_task.py --test_model outputs/train_Baseline-words_fbank8040_LSTMAvg/models/model_100 --txt_name outputs_txts/Baseline-words_fbank8040_LSTMAvg_test_task2.txt --save_path data/trigger_wav/test/task2/ --threshold `python src/get_th.py --total_hours 36.0836 --plt_name  wake_task2.jpg --pkl_names outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task2.pkl --threshold_for_num_false_alarm_per_hour 1.0` --model_class lstm_models --model_name LSTMAvg --word_num 3 --step_size 3 --conf_size 150 --vad_mode 0 --vad_max_length 200 --vad_max_activate 0.8 --predict_length 80 --segment_step 10 --wav_scp_file testset_task2/wav_for_wake.scp


fi


echo "local/run_kws.sh succeeded"

