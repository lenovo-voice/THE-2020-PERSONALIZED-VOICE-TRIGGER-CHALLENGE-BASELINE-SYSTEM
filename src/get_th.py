import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from utils.save_utils import read_pickle
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from utils.file_tool import cnames
import argparse

parser = argparse.ArgumentParser(description = "Show kws system result");
parser.add_argument('--total_hours', type=float,default=20.1309,dest='total_hours',help='Task total time(hours)');
parser.add_argument('--plt_name', type=str,default='',dest='plt_name',help='Figure name');
parser.add_argument('--pkl_names', type=str,default=40,dest='pkl_names',help='Pickle names');
parser.add_argument('--threshold_for_num_false_alarm_per_hour', type=float,default=1.0,dest='th',help='threshold for num false alarm per hour');
args = parser.parse_args();

total_hours = args.total_hours

#total_hours=34.9339 # target
#total_hours = 20.1309
#total_hours = 36.0836

def plot_det(plot_scores, figure_name=None):
    #colors = ["red", "darkorange", "forestgreen", "deepskyblue", "blueviolet"]
    colors = [item for item in cnames.keys()]
    lw = 2
    plt.figure()
    for i, score in enumerate(plot_scores):
        pkl_name = score[0]
        fpr = score[1]
        tpr = score[2]
        #roc_auc = score[3]
        #tnr = fjr = 1 - tpr
        plt.plot(fpr, tpr, color=colors[i*2+10], lw=lw, label='%s' % (pkl_name.split("_")[-2]))

    #plt.plot([0, 1], [1, 0], lw=lw, color='navy', linestyle='--')
    plt.vlines(1, 0, 1.5,lw=lw, color='navy', linestyle='--')
    plt.vlines(2, 0, 0.1,lw=lw, color='red', linestyle='--')
    plt.hlines(0.1, 0,   2,lw=lw, color='red', linestyle='--')
    plt.xlim([0.0, 5.0])
    plt.ylim([0.0, 0.2])
    plt.xlabel('False Alarms Per Hour')
    plt.ylabel('False Rejects')
    #plt.title('Receiver operating characteristic example')
    # plt.legend(bbox_to_anchor=(1,-0.2))
    plt.legend(loc="best")
    plt.grid()
    # plt.axis("equal")
    # plt.tight_layout()
    if figure_name == None:
        plt.show()
    else:
        plt.savefig(figure_name)
        #print("Save Done.")
plt_name = args.plt_name
pkl_names = args.pkl_names.split()
print_th = True
plot_scores = []
for pkl_name in pkl_names:
   fpr = []
   tpr = []
   my_dict = read_pickle(pkl_name)
   y = my_dict["y"]
   scores = my_dict["scores"]
   keys = my_dict["keys"]

   items = zip(y, scores, keys)
   items = sorted(items, key=lambda i:i[1], reverse=True)

   all_right = sum(y)
   #print(sum(scores))
   tp = 0
   fn = 0
   fp = 0
   temp = 0
   for index, item in enumerate(items):
       # print(index, item)
       if item[1] < 0.7:
           temp+=1 
       if item[0] == 1:
           tp += 1
           # if fp != 0:
           #    print("WRONG CASE", item)
           
       if item[0] == 0:
           fp += 1
           #print("ATTENTION! ", fp / total_hours, (all_right - tp) / all_right, index, item)
           # print(fp / total_hours, (all_right - tp) / all_right)
           fpr.append(fp / total_hours)
           tpr.append((all_right - tp) / all_right)
           #break
       if args.th < fp / total_hours < 5 and print_th:
           #print("ATTENTION! ", fp / total_hours, (all_right - tp) / all_right, index, item,pkl_name)
           print(item[1])
           print_th = False
           #break
   plot_scores.append((pkl_name, fpr, tpr))
#print((index+1-temp)/(index+1))

plot_det(plot_scores, plt_name)

