import torch
import torch.nn as nn
from collections import OrderedDict

def returnGPUModel(net, modelName):

    net = nn.DataParallel(net)
    net = net.cuda()

    state = torch.load(modelName)
    net.load_state_dict(state['state_dict'], strict=False)
    print("returnGPUModel")
    return net

def returnCPUModel(net, model_name):

    # load a model trained with gpu
    state = torch.load(model_name, map_location='cpu')['state_dict']
    cpu_model_dict = OrderedDict()

    # create a new dict
    for k, v in state.items():
        name = k[7:] # remove `module.`
        cpu_model_dict[name] = v
    # load the new dict
    net.load_state_dict(cpu_model_dict, strict=False)

    return net

