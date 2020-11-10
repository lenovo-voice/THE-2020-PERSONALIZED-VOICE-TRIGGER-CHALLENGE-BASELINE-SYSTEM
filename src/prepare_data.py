import os
import sys
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description = "Data preparation");
parser.add_argument('--dataset_path', type=str,default=None,dest='dataset_dir',help='Raw dataset path');
parser.add_argument('--dest_path',type=str,default=None,dest='dest_dir',help='Data preparation in Kaldi type');
args = parser.parse_args();

#file_dir = "/Netdata/AudioData/PVTC/official_data/train/"
file_dir = args.dataset_dir
data_dir = args.dest_dir
os.system("mkdir -p "+data_dir)
f1 = open(data_dir + "/text","w")
f2 = open(data_dir + "/wav.scp","w")
f3 = open(data_dir + "/utt2spk","w")
speaker_list = os.listdir(file_dir)
speaker_list.sort()
for spk in tqdm(speaker_list):
    #print("spk:",spk)
    root_dir = os.path.join(file_dir,spk)
    keyword_list = os.listdir(root_dir)
    for keyword in keyword_list:
        #print("keyword:",keyword)
        keyword_dir = os.path.join(root_dir,keyword)
        device_ids = os.listdir(keyword_dir)
        for device in device_ids:
            #print("devices:",device)
            device_dir = os.path.join(keyword_dir,device)
            data = os.listdir(device_dir)
            texts = [item for item in data if ".txt" in item]
            for text in texts:
                text_dir = os.path.join(device_dir,text)
                with open(text_dir) as f:
                    temp = f.readline()
                    f1.writelines(spk+"-"+keyword+"-"+device+"-"+text.replace(".txt",'') + " " + temp + "\n")
                    f2.writelines(spk+"-"+keyword+"-"+device+"-"+text.replace(".txt",'') + " " + os.path.join(device_dir,text.replace(".txt",".wav"))+"\n")
                    f3.writelines(spk+"-"+keyword+"-"+device+"-"+text.replace(".txt",'') + " "+ spk+"\n" )
f1.close()
f2.close()




