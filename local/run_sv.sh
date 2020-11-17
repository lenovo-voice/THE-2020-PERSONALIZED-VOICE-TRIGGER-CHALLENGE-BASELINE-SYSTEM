#!/bin/bash

stage=1
list_pretrain=$1
path_pvtc_train=$2
path_pvtc_dev=$3
musan_path=$4
rir_path=$5
whether_finetune=$6
label="None"

if [ $stage -le 1 ];then
    cd ./sv_part
    python ./dataprep.py --train_set $path_pvtc_train --dev_path $path_pvtc_dev'/task1/wav_data/' --pvtc_trials_path $path_pvtc_dev'/task1/trials' \
        --utt2label $path_pvtc_dev'/task1/trials_for_wake' --split_path '../data/trigger_wav/task1/' --make_sv_trials --make_list|| exit 1
    cd ../
fi

if [ $stage -le 2 ];then
    if [ $list_pretrain = $label ]
    then
    mkdir -p ./sv_part/exps/PVTCpretrain_res34se_asp_sgd/model/
    wget https://github.com/Doctor-Do/PVTC_sv_model/releases/download/11/model000000050.model -O sv_part/exps/PVTCpretrain_res34se_asp_sgd/model/model000000050.model
    else
    cd ./sv_part
    python ./trainSpeakerNet.py --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCpretrain_res34se_asp_sgd' --nClasses 3091 \
        --augment True --n_mels 80 --lr_decay 0.2 --test_interval 15 --lr 0.01 --max_epoch 50\
        --batch_size 256 --scale 32 --margin 0.2 --train_list $list_pretrain --test_list sv_trials --train_path "" \
        --test_path "" --musan_path $musan_path --rir_path $rir_path --optimizer sgd || exit 1
    cd ../
    fi
fi

if [ $stage -le 3 ];then
    if [ $whether_finetune = $label ]
    then
    mkdir -p ./sv_part/exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/
    wget https://github.com/Doctor-Do/PVTC_sv_model/releases/download/fine_tune/model000000020.model -O sv_part/exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model
    else
    cd ./sv_part
	python ./finetune.py --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2' --nClasses 3091 \
    --nClasses_ft 300 --initial_model 'exps/PVTCpretrain_res34se_asp_sgd/model/model000000050.model' --max_epoch 20\
    --augment True --n_mels 80 --lr_decay 0.9 --test_interval 10 --lr 0.001 \
    --batch_size 256 --scale 32 --margin 0.2 --train_list list_pvtc_pure --test_list sv_trials --train_path "" \
    --test_path "" --musan_path $musan_path --rir_path $rir_path --optimizer sgd || exit 1
    cd ../
    fi
fi


if [ $stage -le 4 ];then
    cd ./sv_part
    python ./inference.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task1/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd --devdatapath $path_pvtc_dev'/task1/wav_data/' \
        --trials_list $path_pvtc_dev'/task1/trials' --uttpath '../data/trigger_wav/task1/' --utt2label "../outputs_txts/Baseline-words_fbank8040_LSTMAvg_task1.txt" --u2l_template $path_pvtc_dev'/task1/trials_for_wake' --save_dic True  || exit 1
    cd ../
    cd ./sv_part
    python ./inference.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task2/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd --devdatapath $path_pvtc_dev'/task2/wav_data/'\
        --trials_list $path_pvtc_dev'/task2/trials' --uttpath '../data/trigger_wav/task2/' --utt2label "../outputs_txts/Baseline-words_fbank8040_LSTMAvg_task2.txt"  --u2l_template $path_pvtc_dev'/task2/trials_for_wake' --save_dic True  || exit 1
    cd ../
fi








