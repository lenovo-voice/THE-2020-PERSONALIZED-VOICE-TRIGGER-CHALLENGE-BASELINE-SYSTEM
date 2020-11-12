#!/bin/bash


stage=1


if [ $stage -le 1 ];then
	local/prepare_all.sh /PATH/official_PVTC/train /PATH/official_PVTC/train_xiaole_time_point /PATH/official_PVTC/dev || exit 1
fi

if [ $stage -le 2 ];then
	local/run_kws.sh || exit 1
fi


if [ $stage -le 3 ];then
# 6 parameters in this sh. The first `list_pretrain` needs to be created by yourself based on your pre-training data. More details can be found in ./SV_README.md
# If you set the first `list_pretrain` to None, the pre-trained model we provided will be downloaded and used in next steps.
# The second and third parameters should be the path of PVTC train and dev data.
# The fourth and fifth parameters should be the path of MUSAN(SLR17) and RIRs(SLR28) noise. 
# If the sixth parameter `whether_finetune` set as None, the finetuned model we provided will also be downloaded instead of fine-tuning on the pre-trained model.
	local/run_sv.sh None /PATH/official_PVTC/train /PATH/official_PVTC/dev \
     /PATH/musan/ /PATH/RIRS_NOISES/simulated_rirs/ None || exit 1
fi

local/show_results.sh /PATH/official_PVTC/dev || exit 1

exit 0;


