import torch, torch.nn as nn, torch.nn.functional as F
from torchaudio import transforms
import numpy as np

class StatsPool(nn.Module):
    
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = ConvLayer(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = NormLayer(planes)
#         self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = NormLayer(planes)
#         self.conv3 = ConvLayer(planes, self.expansion*planes, kernel_size=1, bias=False)
#         self.bn3 = NormLayer(self.expansion*planes)
#         self.relu = nn.ReLU(inplace=True)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 ConvLayer(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 NormLayer(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = self.relu(self.bn1(self.conv1(x)))
#         out = self.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = self.relu(out)
#         return out


class ResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, num_classes=10, in_ch=1,is1d=False, **kwargs):
        super(ResNet, self).__init__()
        if is1d:
            self.NormLayer = nn.BatchNorm1d
            self.ConvLayer = nn.Conv1d
        else:
            self.NormLayer = nn.BatchNorm2d
            self.ConvLayer = nn.Conv2d

        self.in_planes = in_planes

        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes*8, num_blocks[3], stride=2)
       

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def ResNet18(in_planes, **kwargs):
    return ResNet(in_planes, BasicBlock, [2,2,2,2], **kwargs)

def ResNet34(in_planes, **kwargs):
    return ResNet(in_planes, BasicBlock, [3,4,6,3], **kwargs)

class ResNet34StatsPool(nn.Module):

    def __init__(self, in_planes, embedding_size, dropout=0.5,n_mels=80, **kwargs):

        super(ResNet34StatsPool, self).__init__()
        self.front = ResNet34(in_planes, **kwargs)
        self.pool = StatsPool()
        self.bottleneck = nn.Linear(in_planes*8*2, embedding_size)
        self.drop = nn.Dropout(dropout) if dropout else None
        self.torchfb = transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160,  n_mels=n_mels)
        
    def forward(self, x):
        x = self.torchfb(x)+1e-6
        x = torch.log(x)
        x = x - x.mean(axis=2).unsqueeze(dim=2)

        x = self.front(x.unsqueeze(dim=1))
        x = self.pool(x)
        x = self.bottleneck(x)
        if self.drop:
            x = self.drop(x)
        return x
# def ResNet50(in_planes, **kwargs):
#     return ResNet(in_planes, Bottleneck, [3,4,6,3], **kwargs)

# def ResNet101(in_planes, **kwargs):
#     return ResNet(in_planes, Bottleneck, [3,4,23,3], **kwargs)

# def ResNet152(in_planes, **kwargs):
#     return ResNet(in_planes, Bottleneck, [3,8,36,3], **kwargs)

def MainModel(nOut=256, **kwargs):
    # Number of filters
    #num_filters = [16, 32, 64, 128]
    in_planes = 32
    model = ResNet34StatsPool(in_planes=in_planes, embedding_size=nOut,**kwargs)
    return model