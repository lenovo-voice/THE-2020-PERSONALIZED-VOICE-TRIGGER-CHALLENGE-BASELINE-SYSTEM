#!/bin/bash

stage=1
list_pretrain=$1
path_pvtc_train=$2
path_pvtc_dev=$3
musan_path=$4
rir_path = $5

if [ $stage -le 1 ];then
    cd ./sv_part
    python ./dataprep.py --train_set $path_pvtc_train --dev_path $path_pvtc_dev'/task1/wav_data/' --pvtc_trials_path $path_pvtc_dev'/task1/trials' \
        --utt2label $path_pvtc_dev'/task1/trials_for_wake'  --make_sv_trials --make_list|| exit 1
    cd ../
fi

if [ $stage -le 2 ];then
    cd ./sv_part
    python ./trainSpeakerNet.py --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCpretrain_res34se_asp_sgd' --nClasses 3091 \
        --augment True --n_mels 80 --lr_decay 0.2 --test_interval 15 --lr 0.01 --max_epoch 50\
        --batch_size 256 --scale 32 --margin 0.2 --train_list $list_pretrain --test_list sv_trials --train_path "" \
        --test_path "" --musan_path $musan_path --rir_path $rir_path --optimizer sgd || exit 1
    cd ../
fi

if [ $stage -le 3 ];then
    cd ./sv_part
	python ./finetune.py --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2' --nClasses 3091 \
    --nClasses_ft 300 --initial_model 'exps/PVTCpretrain_res34se_asp_sgd/model/model000000050.model' --max_epoch 20\
    --augment True --n_mels 80 --lr_decay 0.9 --test_interval 10 --lr 0.001 \
    --batch_size 256 --scale 32 --margin 0.2 --train_list list_pvtc_pure --test_list sv_trials --train_path "" \
    --test_path "" --musan_path $musan_path --rir_path $rir_path --optimizer sgd || exit 1
    cd ../
fi


if [ $stage -le 4 ];then
    cd ./sv_part
    python ./inference.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task1/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd \
        --trials_list $path_pvtc_dev'/task1/trials' --uttpath $path_pvtc_dev'/task1/wav_data/' --utt2label $path_pvtc_dev'/task1/trials_for_wake'  --save_dic True  || exit 1
    cd ../
    cd ./sv_part
    python ./inference.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task2/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000010.model'\
        --scale 32 --margin 0.2  --optimizer sgd \
        --trials_list $path_pvtc_dev'/task2/trials' --uttpath $path_pvtc_dev'/task2/wav_data/' --utt2label $path_pvtc_dev'/task2/trials_for_wake'  --save_dic True  || exit 1
    cd ../
fi








