import os
import sys
import argparse

parser = argparse.ArgumentParser(description = "Prepare positive and negative file");
parser.add_argument('--split_wav_dir', type=str,default=None,dest='split_wav_dir',help='Split wav dir');
parser.add_argument('--data_dir', type=str,default=None,dest='data_dir',help='prepare_data.py destination path');
parser.add_argument('--combine_data_dir', type=str,default=None,dest='combine_data_dir',help='Positive and negative file path');
args = parser.parse_args();

split_wav_dir = args.split_wav_dir

pos_dir = split_wav_dir + "/xiaole/"
neg_dir = split_wav_dir + "/other/"
all_data_dir = args.data_dir
dis_dir = args.combine_data_dir

os.system("mkdir -p "+dis_dir)
pos_list = [[item.replace("小乐小乐","xiaole"),pos_dir+"/"+item] for item in os.listdir(pos_dir)]
neg_list = [[item,neg_dir+"/"+item] for item in os.listdir(neg_dir)]
purn_neg_list = []
for line in open(all_data_dir+"/wav.scp"):
    if "others" in line:
        [utt,wav] = line.strip().split()
        #print(utt)
        purn_neg_list.append(["other_"+utt+".wav "+wav])
all_neg_list = neg_list + purn_neg_list

f1 = open(dis_dir+"/text","w")
f2 = open(dis_dir+"/wav.scp","w")
f3 = open(dis_dir+"/utt2spk","w")
f4 = open(dis_dir+"/neg_wav.scp","w")

for item in pos_list:
    f1.writelines(item[0] + " 小 乐 小 乐\n")
    f2.writelines(" ".join(item) + "\n")
    f3.writelines(item[0] + " " + item[0] + "\n")
for item in all_neg_list:
    f4.writelines(" ".join(item)+"\n")
f1.close();f2.close();f3.close();f4.close()

