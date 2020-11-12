#!/bin/bash

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


nj=40
stage=0
x=PVTC

if [ $stage -le 0 ]; then

utils/fix_data_dir.sh data/$x || exit 1;

steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
      --nj $nj data/$x exp/make_mfcc/ mfcc_hires || exit 1

steps/compute_cmvn_stats.sh data/$x exp/make_mfcc mfcc || exit 1

utils/fix_data_dir.sh data/$x || exit 1

utils/data/limit_feature_dim.sh 0:39 data/$x data/${x}_nopitch || exit 1
    
steps/compute_cmvn_stats.sh data/${x}_nopitch exp/make_mfcc mfcc || exit 1

utils/fix_data_dir.sh data/$x || exit 1;


fi

if [ $stage -le 1 ]; then

steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --use-gpu false --scale_opts "--transition-scale=1.0 --acoustic-scale=10.0 --self-loop-scale=0.1" \
                  data/${x}_nopitch data/lang_aishell exp/nnet3/tdnn_sp/ exp/nnet3_$x || exit 1

fi


if [ $stage -le 2 ]; then

steps/get_train_ctm.sh data/$x data/lang_aishell exp/nnet3_$x || exit 1

fi

exit 0

