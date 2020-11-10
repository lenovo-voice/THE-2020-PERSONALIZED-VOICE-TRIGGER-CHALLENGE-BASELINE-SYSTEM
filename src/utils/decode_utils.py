import math
import torch
import numpy as np
from operator import mul
from functools import reduce


def whether_samespeaker(utt1,utt2,utt3,utt4,threshold):
    with torch.no_grad():
        inp = torch.FloatTensor(loadWAV(utt2wav[utt4],0,True,10)).cuda()
        eval_embd = s.__S__.forward(inp).cpu().numpy()
    enroll_embd = (enroll_dic[utt1] + enroll_dic[utt2] +enroll_dic[utt3]) /3
    result = 1 - spatial.distance.cosine(eval_embd, enroll_embd)
    if result < threshold:
        return False
    else:
        return True



def softmax(x):
    """ softmax """
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

def predict(net, feats):
    """ predict feats """
    torch.set_num_threads(1)
    feats = np.array(feats)
    feats = torch.from_numpy(feats).float()
    #feats = torch.Tensor(feats)
    feats = feats.unsqueeze(1).cpu()
    #print("unsqueeze feat:",feats.shape)
    outputs, _ = net(feats)
    outputs = outputs.cpu().detach().numpy()
    soft_outputs = softmax(outputs)
    # print(soft_outputs.shape, np.argmax(soft_outputs, 1))
    return soft_outputs
def domainemdbegin_predict(net, emd_net, feats):
    """ predict feats """
    torch.set_num_threads(12)
    feats = np.array(feats)
    feats = torch.from_numpy(feats).float()
    feats = feats.unsqueeze(1)
    emd_output, emd = emd_net(feats)
    emd = emd.unsqueeze(1)
    emd = emd.repeat(1,140,1)
    #print(emd.size())
    #print(feats.size())
    outputs, _ = net(feats,emd)
    outputs = outputs.cpu().detach().numpy()
    soft_outputs = softmax(outputs)
    return soft_outputs

def domainemd_predict(net, emd_net, feats):
    """ predict feats """
    torch.set_num_threads(12)
    feats = np.array(feats)
    feats = torch.from_numpy(feats).float()
    feats = feats.unsqueeze(1)
    emd_output, emd = emd_net(feats)
    outputs, _ = net(feats,emd)
    outputs = outputs.cpu().detach().numpy()
    soft_outputs = softmax(outputs)
    return soft_outputs

def domain_predict(net, feats):
    """ predict feats """
    torch.set_num_threads(12)
    feats = np.array(feats)
    feats = torch.from_numpy(feats).float()
    feats = feats.unsqueeze(1)
    outputs, domain = net(feats)
    outputs = outputs.cpu().detach().numpy()
    domain = domain.cpu().detach().numpy()
    soft_outputs = softmax(outputs)
    soft_domain = softmax(domain)
    return soft_outputs,soft_domain


def smooth(scores, smooth_win):
    """scores smooth"""
    scores = np.array(scores)
    smoothed_scores = []
    for i in range(len(scores)):
        cur_score = np.array(scores[max(i-smooth_win+1, 0):i+1,:])
        smoothed_scores.append(np.mean(scores[max(i-smooth_win+1, 0):i+1], 0))
    smoothed_scores = np.array(smoothed_scores)
    return smoothed_scores

def get_full_conf(scores, conf_win, word_num):
    conf_step = 1
    confs = []

    if scores.shape[0] <= conf_win:
        conf = compute_conf2(scores)
        return conf

    for i in range(0, scores.shape[0], conf_step):
        if i + conf_win <= scores.shape[0]:
            c = compute_conf2(scores[i:i+conf_win], word_num)
        else:
            c = compute_conf2(scores[scores.shape[0]-conf_win:scores.shape[0]], word_num)
        confs.append(c)
  
    return max(confs)

def compute_conf2(scores, word_num=3):
    scores = scores[:, 1:]
    h = np.zeros(scores.shape)
    M = scores.shape[1]
    Ts = scores.shape[0]
    # compute score for the first keyword
    h[0][0] = scores[0][0]
    for i in range(1, Ts):
        h[i][0] = max(h[i - 1][0], scores[i][0])
    # computing score for the remaining keywords
    for k in range(1, M):
        h[k][k] = h[k - 1][k - 1] * scores[k][k]
        for t in range(k + 1, Ts):
            h[t][k] = max(h[t - 1][k], h[t - 1][k - 1] * scores[t][k])

    return h[Ts - 1][M - 1] ** (1/word_num)


def get_full_conf2(scores, conf_win, word_num):
    """ Sliding window to get confidence """
    conf_step = 1
    confs = []

    if scores.shape[0] <= conf_win:
        conf, max_id = compute_conf4(scores)
        return conf, max_id

    cur_max_c = 0
    cur_max_id = None
    for i in range(0, scores.shape[0], conf_step):
        if i + conf_win <= scores.shape[0]:
            c, max_id = compute_conf4(scores[i:i+conf_win], word_num)
        else:
            c, max_id = compute_conf4(scores[scores.shape[0]-conf_win:scores.shape[0]], word_num)
        if c > cur_max_c:
            cur_max_c = c
            cur_max_id = max_id

    # print(cur_max_c, cur_max_id)
    return cur_max_c, cur_max_id

def compute_conf4(scores, word_num=3):
    scores = scores[:, 1:]
    max_id = np.argmax(scores, 0)
    scores = np.max(scores, 0)
    scores = np.prod(scores)
    return scores ** (1/word_num), max_id

