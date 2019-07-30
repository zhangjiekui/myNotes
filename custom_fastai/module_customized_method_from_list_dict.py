# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-7-30 16:57
# Project: 00codes
# IDE:PyCharm
# torch.set_printoptions(linewidth=300)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net=Net()
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)
# net.to(device)


from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import torch.nn as nn


class DummyModule():
    def __init__(self, n_in, nh, n_out):
        self._modules = {}
        self.l1 = nn.Linear(n_in, nh)
        self.l2 = nn.Linear(nh, n_out)

    def __setattr__(self, k, v):
        if not k.startswith("_"): self._modules[k] = v
        super().__setattr__(k, v)

    def __repr__(self):
        return f'{self._modules}'

    def parameters(self):
        for l in self._modules.values():
            for p in l.parameters(): yield p

# class DummyModule1():
#     def __init__(self, n_in, nh, n_out):
#         self._m = {}
#         self.l1 = nn.Linear(n_in, nh)
#         self.l2 = nn.Linear(nh, n_out)
#         # print("ok",self._m)
#
#     def __setattr__(self, key, value):
#         # print(key)
#         if not key.startswith('_'):
#             # print(self._m)
#             self._m[key] = value
#         super().__setattr__(key, value)
#
#             # print(f"key not startswith('_')={key},value={value}")
#             # print(type(super()).__name__)
#
#     def __repr__(self):
#         return f"{self._m}"
#
#     def parameters(self):
#         for l in self._m.values():
#             for p in l.parameters():
#                 yield p


class DummyModule1():
    def __init__(self, n_in, nh, n_out):
        self._m = {}
        self.l1 = nn.Linear(n_in, nh)
        self.l2 = nn.Linear(nh, n_out)
        print("ok", self._m)

    def __setattr__(self, key, value):
        print(key)
        if not key.startswith('_'):
            print(f"key not startswith('_')={key},value={value}")
            self._m[key] = value
        super().__setattr__(key, value)
        print(type(super()).__name__)

    def __repr__(self):
        return f"{self._m}"

    def parameters(self):
        for l in self._m.values():
            for p in l.parameters():
                yield p

# 登记模块 Registering modules
class ModuleReg(nn.Module):
    def __init__(self,layers,l_names):
        super().__init__()
        self.layers=layers
        self.l_names=l_names
        for i,l in enumerate(layers):
            self.add_module(self.l_names[i],l)
            self.__setattr__(self.l_names[i],l)

    def __call__(self, x):
        for l in self.layers:
            x=l(x)




if __name__ == '__main__':
    mdl = DummyModule(10, 20, 1)
    print(mdl)
    # mdl1 = DummyModule1(784, 50, 10)
    # ps=[o.shape for o in mdl1.parameters()]
    # print(ps)
    # print(mdl1)
    m, nh, c= 784, 50, 10
    layers = [nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, c)]
    l_names=["l1_zhang1","l2_relu2","l3_zhao3"]

    mr=ModuleReg(layers,l_names)
    print("new:",mr)
    print("named_children()",list(mr.named_children()))
    # print(list(mdl.named_children()))

    module_dict={}
    layers = [nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, c)]
    l_names=["l1_zhang1","l2_relu2","l3_zhao3"]
    for i, x_layer in enumerate(layers):
        module_dict[l_names[i]]=x_layer

    print(module_dict)
    print(module_dict.values())

    print("model = nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, 10))-----------")
    model = nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, 10))
    print("model:",model)

    layers_list = [nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, c),nn.LSTM(m, nh)]
    model_from_list=nn.Sequential(*layers_list)
    print(model_from_list)




    pass


