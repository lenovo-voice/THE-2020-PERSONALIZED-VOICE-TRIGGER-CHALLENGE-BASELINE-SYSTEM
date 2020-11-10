#!/bin/bash


stage=1


if [ $stage -le 1 ];then
	local/prepare_all.sh /Netdata/AudioData/PVTC/official_data/train /Netdata/AudioData/PVTC/official_data/new_xiaole_split /Netdata/AudioData/PVTC/official_data/dev|| exit 1
fi

if [ $stage -le 2 ];then
	local/run_kws.sh || exit 1
fi

if [ $stage -le 3 ];then
# pretrian list needs to be builded by yourself.
	local/run_sv.sh ./list_pretrain /Netdata/AudioData/PVTC/official_data/train/ /Netdata/AudioData/PVTC/official_data/dev/ /home/caidanwei/musan/ /home/caidanwei/RIRS_NOISES/simulated_rirs/ || exit 1
fi

local/show_results.sh /Netdata/AudioData/PVTC/official_data/dev/ || exit 1

exit 0;






