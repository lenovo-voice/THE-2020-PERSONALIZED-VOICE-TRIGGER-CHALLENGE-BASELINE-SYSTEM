from utils.save_utils import read_pickle
import sys
import argparse

parser = argparse.ArgumentParser(description = "SpeakerNet");
parser.add_argument('--pickle_name',type=str,default='outputs_pkls/Baseline-words_fbank8040_LSTMAvg_task1.pkl',dest='pkl_name',help='Saving pickle file name')
parser.add_argument('--txt_name',type=str,default='outputs_txts/Baseline-words_fbank8040_LSTMAvg_task1.txt',dest='txt_name',help='Saving txt file name')
parser.add_argument('--threshold',type=float,default=None,dest='th',help='Threshold')
args = parser.parse_args();


pkl_name = args.pkl_name
dist_name = args.txt_name
th = args.th   
raw_pkl = read_pickle(pkl_name)
count = 0
sum_item = 0
f1 = open(dist_name,"w")
for index,utt in enumerate(raw_pkl["keys"]):
    scores = raw_pkl["scores"][index]
    if scores > th:
        judge = "trigger"
    else:
        judge = "non-trigger"
    f1.writelines(utt + " " + judge + " " + str(raw_pkl["time"][index]) + "\n")
f1.close()

