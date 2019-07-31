# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-7-31 15:02
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
import torch
import numpy as np

class Dataset():
    def __init__(self,xs,ys):
        assert len(xs)==len(ys)
        self.xs=xs
        self.ys=ys
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, i):
        return self.xs[i],self.ys[i]

class Sampler():
    def __init__(self,dataset,batchsize,shuffle=False):
        self.n=len(dataset)
        self.batchsize=batchsize
        self.shuffle=shuffle
    def __iter__(self):
        idxs=torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0,self.n,self.batchsize):
            yield idxs[i:i+self.batchsize]

def collate(b):
    # b=map(torch.tensor,b)
    xs,ys=zip(*b)
    xstack=torch.stack(xs)
    ystack=torch.stack(ys)
    return xstack,ystack


class Dataloader():
    def __init__(self,dataset,sampler,collat_fn=collate):
        self.dataset,self.sampler,self.collat_fn=dataset, sampler, collat_fn
    def __iter__(self):
        for s in self.sampler:
            # data_items=[self.dataset[i] for i in s]
            # yield data_items
            yield self.collat_fn([self.dataset[i] for i in s])

if __name__ == '__main__':
    xs=torch.arange(10)
    ys=torch.arange(10,20)
    print(xs)
    print(ys)
    ds=Dataset(xs,ys)
    sampler=Sampler(ds,3,shuffle=True)
    print(list(sampler))


    dl=Dataloader(ds,sampler,collat_fn=collate)
    one_batch=next(iter(dl))
    print(one_batch)

    print("====="*10)

    import pickle,gzip
    from fastai import datasets
    # import fastai.datasets as datasets
    from torch import tensor


    MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
    def get_data():
        path = datasets.download_data(MNIST_URL, ext='.gz')
        with gzip.open(path, 'rb') as f:
            ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
        return map(tensor, (x_train, y_train, x_valid, y_valid))


    x_train, y_train, *_ = get_data()
    print("x,y shape:",x_train.shape,y_train.shape)
    m_dataset=Dataset(x_train,y_train)
    m_sampler=Sampler(m_dataset,640,shuffle=True)
    m_dataloader=Dataloader(m_dataset,sampler,collat_fn=collate)
    ms,my=next(iter(m_dataloader))
    print("m_xs",ms.shape)
    print("y_xs", my.shape)

    pass


