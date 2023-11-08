import torch
from msadapter.tools import pth2ckpt
from mindspore.common.tensor import Tensor
import msadapter.pytorch as msatorch

import numpy

model_path = '../ckpt/cifar10/corruptions/Standard.pt'
standard_dict = torch.load(model_path, map_location=torch.device('cpu'))
standard_dict = standard_dict['state_dict']
for k, v in standard_dict.items():
    v = Tensor(v.numpy())
    print(type(v))


msatorch.save(standard_dict, '../ckpt/cifar10/corruptions/Standard_state_dict.ckpt')


# print(torch.load('ckpt/cifar10/corruptions/Standard_state_dict.pt'))
state_dict = msatorch.load('../ckpt/cifar10/corruptions/Standard_state_dict.ckpt')
print(state_dict)