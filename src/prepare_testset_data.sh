#!/bin/bash

test_set=$1
dest_dir=$2

mkdir -p $dest_dir
awk '{printf("%s\n%s\n%s\n",$1,$2,$3)}' $test_set/trials_competitor | sort -u  | awk '{printf("%s %s/wav_data/%s\n",$1,test_set,$1)}' test_set=$test_set  > $dest_dir/wav_for_sv.scp

awk '{printf("%s\n",$4)}' $test_set/trials_competitor | sort -u  | awk '{printf("%s %s/wav_data/%s\n",$1,test_set,$1)}' test_set=$test_set  > $dest_dir/wav_for_wake.scp

cp $test_set/trials_competitor $dest_dir/


