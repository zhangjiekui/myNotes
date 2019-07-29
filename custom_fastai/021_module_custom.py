# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-7-29 08:58
# Project: 00codes
# IDE:PyCharm


from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import torch
from fastai.basics import *
from fastai import datasets

class Module():
    def __init__(self,obj_name):
        self.obj_name=obj_name
        print(f"Initialization: obj_name={self.obj_name}")
    def __call__(self, *args, **kwargs):
        self.args=args
        self.out=self.forward(*self.args)
        print(f"Forward: class_name={type(self).__name__}, obj_name={self.obj_name}, n_args={len(self.args)}")
        return self.out
    def forward(self):
        raise Exception("Error! forward() method Not Implemented!")
    def backward(self):
        print(f"Backward: class_name={type(self).__name__}, obj_name={self.obj_name}, n_args={len(self.args)}")
        self.bwd(self.out,*self.args)

class Relu(Module):
    def forward(self,inp):
        return inp.clamp_min(0.)-0.5
    def bwd(self,out,inp):
        inp.g=(inp>0).float()*out.g

class Lin(Module):
    def __init__(self,obj_name,w,b):
        super(Lin,self).__init__(obj_name)
        self.w=w
        self.b=b
    def forward(self,inp):return inp@self.w+self.b
    def bwd(self,out,inp):
        inp.g=out.g@self.w.t()
        self.w.g=torch.einsum('bi,bj->ij',inp,out.g)
        self.b.g=out.g.sum(0)
class Mse(Module):
    def forward(self,inp,target):return (inp.squeeze() - target).pow(2).mean()
    def bwd(self,out,inp,traget):inp.g = 2*(inp.squeeze()-traget).unsqueeze(-1) / traget.shape[0]

class Model():
    def __init__(self,layers):
        self.layers=layers
        self.loss=Mse("mse_loss")
    def __call__(self,x,target):
        for l in self.layers:
            x=l(x)
        return self.loss(x,target)
    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers):
            l.backward()

if __name__ == '__main__':
    MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl'


    def get_data():
        path = datasets.download_data(MNIST_URL, ext='.gz')
        with gzip.open(path, mode='rb') as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
            return map(tensor, (x_train, y_train, x_valid, y_valid))


    def normalize(x, mean, std):
        return (x - mean) / std


    x_train, y_train, x_valid, y_valid = get_data()
    mean = x_train.mean()
    std = x_train.std()
    x_train_nor = normalize(x_train, mean, std)
    # Note: 要使用训练集的mean、std 来标准化验证集
    x_valid_nor = normalize(x_valid, mean, std)
    n, m = x_train.shape
    c = y_train.max() + 1
    n, m, c, c.item()
    nh = 50
    w1 = torch.randn(m, nh) * math.sqrt(2. / m)  ## kaiming init / he init for
    b1 = torch.zeros(nh)
    w2 = torch.randn(nh, 1) / math.sqrt(nh)
    b2 = torch.zeros(1)

    # relu2=Relu("relu2")
    # lin1=Lin('lin1_50',1,0)
    # lin3 = Lin('lin3_1', 1, 0)

    w1.g, b1.g, w2.g, b2.g = [None] * 4
    lin1_50 = Lin("lin1_50",w1, b1)
    rel_2 = Relu("relu2")
    lin2_3 = Lin("lin3_1",w2, b2)
    layers = [lin1_50, rel_2, lin2_3]
    model2 = Model(layers)
    loss2 = model2(x_train_nor, y_train.float())
    model2.backward()
    pass


