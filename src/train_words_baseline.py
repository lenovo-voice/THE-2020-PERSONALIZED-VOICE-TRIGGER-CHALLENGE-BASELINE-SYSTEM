import os
import math
import torch
import scipy
import random
import datetime
import scipy.stats
import numpy as np
import torch.nn as nn
import soundfile as sf
import torch.optim as optim
import torch.nn.functional as F
import importlib
from torch.autograd import Variable
from sklearn.metrics import recall_score, confusion_matrix
from tqdm import tqdm
from sklearn.metrics import roc_auc_score 
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from dataloader_phone import NpyDataset,MyDataLoader
import argparse

parser = argparse.ArgumentParser(description = "Model training");
parser.add_argument('--seed', type=int,default=40,dest='seed',help='Random seed');
parser.add_argument('--mode', type=str,default='train',dest='MODE',help='train resume val');
parser.add_argument('--task_name',type=str,default='Baseline-word',dest='task_name',help='Task name');
parser.add_argument('--model_class',type=str,default='lstm_models',dest='model_class',help='Model class');
parser.add_argument('--model_name',type=str,default='LSTMAvg',dest='model_name',help='Model name');
parser.add_argument('--index_dir',type=str,default='./index_words/',dest='index_root_dir',help='Training set path');
parser.add_argument('--batch_size',type=int,default=128,dest='BATCH_SIZE',help='Batch size');
parser.add_argument('--num_epoch',type=int,default=100,dest='EPOCH_NUM',help='Number of epochs');
parser.add_argument('--lr', type=float, default=0.01,dest='LEARNING_RATE', help='Learning rate');
parser.add_argument('--resume_model', type=str, default=None,dest='RESUME_MODEL_NAME', help='Resume model path');
parser.add_argument('--resume_lr', type=float, default=0.0001,dest='RESUME_LR', help='Resume learning rate');
args = parser.parse_args();




seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# type

task_name = args.task_name
feat_name = "fbank8040"
model_name = args.model_name
# record
MODE = args.MODE
WORKSPACE_PATH = "./"
NAME_SPACE = MODE + "_" + task_name + "_" + feat_name + "_" + model_name
if not os.path.exists(WORKSPACE_PATH + "/outputs/" + NAME_SPACE + "/models/"):
    os.makedirs(WORKSPACE_PATH + "/outputs/" + NAME_SPACE + "/models/")
    os.makedirs(WORKSPACE_PATH + "/outputs/" + NAME_SPACE + "/log/")

SAVE_PATH = os.path.join(WORKSPACE_PATH, "outputs", NAME_SPACE, "models")
LOG_PATH = os.path.join(WORKSPACE_PATH, "outputs", NAME_SPACE, "log")

LOG_FILE = LOG_PATH + "/" + NAME_SPACE + ".log"
f_log = open(LOG_FILE, "wt")

# config
WIN_SIZE= 40
EPOCH_NUM = args.EPOCH_NUM
BATCH_SIZE = args.BATCH_SIZE
LEARNING_RATE = args.LEARNING_RATE
RESUME_MODEL_NAME = args.RESUME_MODEL_NAME # IMFCC 
RESUME_LR = args.RESUME_LR

# data
index_root_dir = args.index_root_dir + '/'
train_utt2data = [ [line.split()[0], line.split()[1] ] for line in open(index_root_dir + 'train_utt2npy') ]

label2int = dict([line.split() for line in open(index_root_dir + 'label2int')])
dev_utt2data = [ [line.split()[0], line.split()[1] ] for line in open(index_root_dir + 'dev_utt2npy') ]
utt2label = dict([line.split() for line in open(index_root_dir + 'utt2label')])

# net
net = importlib.import_module('models.'+args.model_class).__getattribute__(args.model_name)(output_dim=len(label2int))
#net = LSTMAvg(output_dim=len(label2int))
print(net)
net = nn.DataParallel(net)
net = net.cuda()
criterion = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True, min_lr=1e-4)

def saveModel(epoch, temp=""):
    global net
    global optimizer

    now = datetime.datetime.now()
    time_str = now.strftime('%Y_%m_%d_%H')
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, SAVE_PATH + "/" +  "model_" + str(epoch))


def batch_handle(feats, targets):
    input_x = torch.unsqueeze(feats, dim=1).float().cuda()
    targets = targets.cuda()
    return input_x, targets


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(epc = 1):
    global net
    global optimizer
    global criterion

    # data loader preparation

    train_dataset = NpyDataset(train_utt2data, utt2label, label2int, need_aug=True, with_label=True, shuffle=True, win_size=WIN_SIZE)
    dear_train_dataloader = MyDataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=20)

    net.train()

    iteration_num = 0
    correct = 0
    losses = AverageMeter()

    print('Training...')
    for batch_utt, batch_x, batch_y in tqdm(dear_train_dataloader, total=len(dear_train_dataloader)):
        iteration_num += 1
        batch_x, batch_y = batch_handle(batch_x, batch_y)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs, outputs2 = net(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # updata loss
        losses.update(loss.data, batch_x.size()[0])

        if iteration_num % 30 == 29:    
            _, pred = torch.max(outputs, 1) # .data.max(1)[1] # get the index of the max log-probability
            correct = pred.eq(batch_y.data.view_as(pred)).long().cpu().sum()
            curr_log = '[%d, %5d] loss: %.3f, acc: %d / %d. \n' % (epc, iteration_num + 1, losses.avg, correct, BATCH_SIZE)
            tqdm.write(curr_log)
            f_log.write(curr_log)
        #time_start=time.time()
    return losses.avg

def validate(epoch):
    global net

    print("Train data set prepared...")
    dev_dataset = NpyDataset(dev_utt2data, utt2label, label2int, need_aug=False, with_label=True, shuffle=False, win_size=WIN_SIZE)
    dear_dev_dataloader = MyDataLoader(dev_dataset, batch_size=BATCH_SIZE, num_workers=12)

    losses = AverageMeter()
    net.eval()
    correct = 0
    scores = []
    # confusion matrix
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (batch_utt, batch_x, batch_y) in tqdm(dear_dev_dataloader, total=len(dear_dev_dataloader)):
            batch_x, batch_y = batch_handle(batch_x, batch_y)
            outputs, _ = net(batch_x)
            test_loss = F.nll_loss(F.log_softmax(outputs, dim=1), batch_y).item()
            losses.update(test_loss, batch_x.size()[0])
            _, pred = torch.max(outputs, 1) # get the index of the max log-probability
            correct += pred.eq(batch_y.data.view_as(pred)).long().cpu().sum().numpy()
            y_true.append(batch_y.long().cpu())
            y_pred.append(pred.long().cpu())    
            

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    print(confusion_matrix(y_true, y_pred))
    curr_log = 'Test set: Loss: {:.3f}, Accuracy: {}/{} ({:.0f}%), UAR: {}.\n'.format(
        losses.avg, correct, len(dear_dev_dataloader.dataset),
        100. * correct / len(dear_dev_dataloader.dataset), recall_score(y_pred, y_true, average='weighted'))
    print(curr_log)
    f_log.write(curr_log)

    return losses.avg

def resume(modelName, lr):
    global net
    global optimizer
    global MODE

    state = torch.load(modelName)
    net.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if MODE == "resume":
        main(state['epoch'] + 1)
    elif MODE == "val":
        validate(state['epoch'])


def returnModel(modelName):
    global net
    state = torch.load(modelName)
    net.load_state_dict(state['state_dict'])
    return net


def main(epc = 1):
    val_losses_avg = 9999
    for epoch in range(epc, EPOCH_NUM + 1):
        print("Current running model [ " + NAME_SPACE + " ]")
        losses_avg = train(epoch)
        scheduler.step(losses_avg)
        if epoch in [1, 3, 5] or epoch % 5 == 0: # epoch in [1, 3, 5, 10, 20, 30, 40, 50]:
            cur_val_losses_avg = validate(epoch)
            saveModel(epoch)
    f_log.close()

 
if MODE == "train":
    main()
elif MODE == "resume":
    resume(RESUME_MODEL_NAME, RESUME_LR)
elif MODE == "val":
    resume(RESUME_MODEL_NAME, RESUME_LR)
else:
    print("Getting model...")


