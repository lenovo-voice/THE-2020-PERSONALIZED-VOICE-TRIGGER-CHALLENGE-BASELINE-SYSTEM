import os
import sys
import random
import argparse

parser = argparse.ArgumentParser(description = "Task data preparation");
parser.add_argument('--dev_dataset', type=str,default=None,dest='dev_dataset',help='Raw dataset path');
parser.add_argument('--dest_dir', type=str,default=None,dest='dest_dir',help='Data preparation in Kaldi type');
args = parser.parse_args();

dev_dir = args.dev_dataset
dis_dir = args.dest_dir
os.system("mkdir -p "+dis_dir)
task_data = os.listdir(dev_dir)

for task_name in task_data:
    task_dir = os.path.join(dev_dir,task_name)
    trails = dict(item.strip().split() for item in open(os.path.join(task_dir,"trials_for_wake")))
    os.system("mkdir -p "+os.path.join(dis_dir,task_name))
    f1 = open(os.path.join(dis_dir,task_name,"utt2wav"),"w")
    f2 = open(os.path.join(dis_dir,task_name,"utt2label"),"w")
    f3 = open(os.path.join(dis_dir,task_name,"label2int"),"w")
    for key in trails.keys():
        f1.writelines(key + " " + os.path.join(task_dir,"wav_data",key)+"\n")
        f2.writelines(key + " "+trails[key]  + "\n")
    f3.writelines("negative 0\npositive 1\n")
    f1.close();f2.close();f3.close()


