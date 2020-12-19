#!/bin/bash

stage=1
path_pvtc_dev=$1


if [ $stage -le 1 ];then
    cd ./sv_part
    python ./compute_score.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task1/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd --alpha 1,1.5,2,3,5,10,15,19,20\
        --trials_list $path_pvtc_dev'/task1/trials' --uttpath '../data/trigger_wav/task1/' --utt2label "../outputs_txts/Baseline-words_fbank8040_LSTMAvg_task1.txt"  --save_dic False \
        --parameter_savepath "eer_threshold.npy"   || exit 1
    cd ../

    cd ./sv_part
    python ./compute_score.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task2/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd --alpha 1,1.5,2,3,5,10,15,19,20\
        --trials_list $path_pvtc_dev'/task2/trials' --uttpath '../data/trigger_wav/task2/' --utt2label "../outputs_txts/Baseline-words_fbank8040_LSTMAvg_task2.txt"  --save_dic False \
        --parameter_savepath "eer_threshold.npy"  || exit 1
    cd ../

    cd ./sv_part
    python ./compute_score.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task1/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd --alpha 1,1.5,2,3,5,10,15,19,20 --utt2wav '../testset_task1/wav_for_sv.scp'\
        --trials_list '../testset_task1/trials_competitor' --uttpath '../data/trigger_wav/test/task1/' --utt2label "../outputs_txts/Baseline-words_fbank8040_LSTMAvg_test_task1.txt"  --save_dic False \
        --parameter_savepath "eer_threshold.npy"   || exit 1
    cd ../

    cd ./sv_part
    python ./compute_score.py --inference  --model ResNetSE34v2 --log_input True --encoder_type ASP --trainfunc amsoftmax --save_path 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/result/task1/' --nClasses 300 \
        --augment True --n_mels 80 --lr_decay 0.2  --lr 0.01  --initial_model 'exps/PVTCfinetune_res34se_asp_sgd_pure_v2/model/model000000020.model'\
        --scale 32 --margin 0.2  --optimizer sgd --alpha 1,1.5,2,3,5,10,15,19,20 --utt2wav '../testset_task2/wav_for_sv.scp'\
        --trials_list '../testset_task2/trials_competitor' --uttpath '../data/trigger_wav/test/task2/' --utt2label "../outputs_txts/Baseline-words_fbank8040_LSTMAvg_test_task2.txt"  --save_dic False \
        --parameter_savepath "eer_threshold.npy"   || exit 1
    cd ../
fi

if [ $stage -le 2 ];then
    python compute_final_score.py --trials $path_pvtc_dev'/task1/trials' --score_files 'Baseline_task1_01.txt'
    python compute_final_score.py --trials $path_pvtc_dev'/task2/trials' --score_files 'Baseline_task2_01.txt'
fi
