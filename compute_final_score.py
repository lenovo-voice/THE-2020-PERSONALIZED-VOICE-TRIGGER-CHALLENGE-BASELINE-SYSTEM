import argparse
import sys


parser = argparse.ArgumentParser(description = "Compute final score");
parser.add_argument('--trials', type=str,default=None,dest='trials',help='Official trials');
parser.add_argument('--score_files', type=str,default=None,dest='score_files',help='Score files submitted');
args = parser.parse_args();

def main(labels, scores, alpha):
    count_pos = 0
    count_neg = 0
    miss = 0
    fa = 0
    for utts in labels.keys():
        i = labels[utts]
        if i == 'positive':
            count_pos = count_pos+1
        else:
            count_neg = count_neg+1
    for utts in scores.keys():
        score = scores[utts]
        assert score in ["0","1"]
        assert labels[utts] in ["positive","negative"]
        if score=='0' and labels[utts]=='positive':
            miss = miss+1
        elif score =='1' and labels[utts]=='negative':
            fa = fa+1
    miss_rate = miss / count_pos
    far = fa / count_neg
    print("Final score:",miss_rate+ alpha*far,"False alarm:",fa / count_neg,"Miss:",miss_rate)
    return miss_rate+ alpha*far , far, miss_rate

def check_scores(trials,scores):
    assert (len(trials.keys()) == len(scores.keys())), "ERROR:Score files length is not equ to Official trials"
    for key in trials.keys():
        assert key in scores.keys(), "ERROR:{0} is not in score files".format(key)
    return True

trials = {' '.join(line.strip().split()[:4]):line.strip().split()[-1] for line in open(args.trials)}
scores = {' '.join(line.strip().split()[:4]):line.strip().split()[-1] for line in open(args.score_files)}

if check_scores(trials,scores):
    main(trials,scores,19)



