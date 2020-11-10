import os
from tqdm import tqdm
import random
import sys
import argparse
parser = argparse.ArgumentParser(description = "Neural network dataset preparation");
parser.add_argument('--pos_feat_dir', type=str,default=None,dest='pos_feat_dir',help='Positive features path');
parser.add_argument('--neg_feat_dir', type=str,default=None,dest='neg_feat_dir',help='Negative features path');
parser.add_argument('--dest_dir', type=str,default=None,dest='dest_dir',help='Destination path');
args = parser.parse_args();

seed=40
random.seed(seed)

pos_dir = args.pos_feat_dir + "/"
neg_dir = args.neg_feat_dir + "/"

root_dir = args.dest_dir

os.system("mkdir -p "+root_dir)


def writelistfile(filename,dir):
    f1 = open(filename,"w")
    data = os.listdir(dir)
    for item in tqdm(data):
        f1.writelines(item + "\n")
    f1.close()
writelistfile(root_dir + "/pos_list",pos_dir)
writelistfile(root_dir + "/neg_list",neg_dir)


spk = [item.split('-')[0].split("_")[-1] for item in open(root_dir + "/pos_list")]
print(spk[0])
spk = list(set(spk))

train_spk = spk[:290]
dev_spk = spk[290:300]

print("pos_list speaker number:",len(spk))
print("train speaker number:",len(train_spk))
print("dev speaker number:",len(dev_spk))

f_spk2id = open(root_dir + "/train_spk2int","w")
f_spk2id_test = open(root_dir + "/dev_spk2int","w")
data_dict = {}

spk_id = 0
for spk_item in train_spk:
    f_spk2id.writelines(spk_item + " " + str(spk_id) + "\n")
    spk_id += 1
f_spk2id.close()

spk_id = 0
for spk_item in dev_spk:
    f_spk2id_test.writelines(spk_item + " " + str(spk_id) + "\n")
    spk_id += 1
f_spk2id_test.close()

for item in open(root_dir + "/pos_list"):
    item = item.strip()
    spk_item = item.split('-')[0].split("_")[-1]
    if spk_item in spk:
        if spk_item not in data_dict.keys():
            data_dict[spk_item] = []
        data_dict[spk_item].append(item)

neg_data_dict = {}
for item in open(root_dir + "/neg_list"):
    item = item.strip()
    spk_item = item.split('-')[0].split("_")[-1]
    if spk_item in spk:
        if spk_item not in neg_data_dict.keys():
            neg_data_dict[spk_item] = []
        neg_data_dict[spk_item].append(item)
    else:
        continue
    
train_utt2wav = []
dev_utt2wav = []
test_utt2wav = []

f5 = open(root_dir + "/utt2label","w")
f6 = open(root_dir + "/utt2spk","w")

pos_list = [item.strip() for item in open(root_dir + "/pos_list")]

for spk in tqdm(data_dict.keys()):
    utt_list = data_dict[spk]
    for index,utt in enumerate(utt_list):
        if spk in train_spk:
            train_utt2wav.append(utt + ' ' + pos_dir + utt +  '\n')
        elif spk in dev_spk:
            dev_utt2wav.append(utt + ' ' + pos_dir + utt + '\n')
        else:
            continue
        f5.writelines(utt + " " + utt.split("_")[0]  + '\n')
        f6.writelines(utt + " " + spk + "\n")        

print("train pos wav length:",len(train_utt2wav))
print("dev pos wav length:",len(dev_utt2wav))

neg_list = [item.strip() for item in open(root_dir + "/neg_list")] 

i=0;j=0

for spk in tqdm(neg_data_dict.keys()):
    utt_list = neg_data_dict[spk]
    for index,utt in enumerate(utt_list):
        if spk in train_spk:
            train_utt2wav.append(utt + ' ' + neg_dir + utt +  '\n')
            i += 1
        elif spk in dev_spk:
            dev_utt2wav.append(utt + ' ' + neg_dir + utt + '\n')
            j += 1
        else:
            continue
        f5.writelines(utt + " filler\n")
        f6.writelines(utt + " " + spk + "\n")

print("train neg wav length:",i)
print("dev neg wav length:",j)

random.shuffle(train_utt2wav);random.shuffle(dev_utt2wav);random.shuffle(test_utt2wav)

f5.close()

f1 = open(root_dir + "/train_utt2npy","w");f2 = open(root_dir + "/dev_utt2npy","w")

def writefile(data,f):
    for item in tqdm(data):
        f.writelines(item)
    f.close()

writefile(train_utt2wav,f1)
writefile(dev_utt2wav,f2)

with open(root_dir + "/label2int","w") as f:
    f.writelines("filler 0\nxiaole 1\nlexiao# 2\nxiao#le# 3")

