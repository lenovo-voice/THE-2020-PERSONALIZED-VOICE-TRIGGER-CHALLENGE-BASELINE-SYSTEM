#!/bin/bash


stage=1


if [ $stage -le 1 ];then
	local/prepare_all.sh /PATH/official_PVTC/train /PATH/official_PVTC/train_xiaole_time_point /PATH/official_PVTC/dev || exit 1
fi

if [ $stage -le 2 ];then
	local/run_kws.sh || exit 1
fi

if [ $stage -le 3 ];then
# pretrian list needs to be builded by yourself.
	local/run_sv.sh ./list_pretrain /PATH/official_PVTC/train /PATH/official_PVTC/dev /PATH/musan/ /PATH/simulated_rirs/ || exit 1
fi

local/show_results.sh /PATH/official_PVTC/dev || exit 1

exit 0;






