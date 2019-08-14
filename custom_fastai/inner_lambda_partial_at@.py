# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-8-14 12:33
# Project: 00codes
# IDE:PyCharm
# torch.set_printoptions(linewidth=300,precision=4,sci_mode=False)
# torch.Tensor.ndim = property(lambda x: len(x.shape))
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

torch.Tensor.ndim = property(lambda x: len(x.shape))
torch.set_printoptions(linewidth=300, precision=4, sci_mode=False)


# 业务需要两个甚至多个参数，但函数定义只接受一个参数

# 函数定义只接受一个参数
from time import sleep
def slow_calculation(cb=None):
    res = 0
    for i in range(2):
        res += i * i
        sleep(1)
        if cb: cb(i)
    return res


# 业务需要两个甚至多个参数
def show_progress(exclamation: str, epoch):  # exclamation:感叹语
    print(f" {exclamation}{epoch}")


# 三种解决办法
def show_progress(exclamation: str, epoch):  # exclamation:感叹语
    print(f"{exclamation} {epoch} ")


def inner_show_progress(exclamation):
    def _inner_function(epoch):
        print(exclamation, epoch)

    return _inner_function


def lam_show_progress(exclamation):
    _inner_lambda = lambda epoch: print(f"{exclamation}{epoch}")
    return _inner_lambda


from functools import partial

_partial_func = partial(show_progress, "使用partial函数：")

_inner_func = inner_show_progress("使用内部函数：")
_inner_lamfunc = lam_show_progress("使用lambda函数：")
# -----------------------------------------------------

slow_calculation(_partial_func)
slow_calculation(_inner_func)
slow_calculation(_inner_lamfunc)


# 使用@函数方式：
def show_epochs(f):
    def _inner(epoch): return partial(f, epoch)

    return _inner


@show_epochs
def show_progress(exclamation: str, epoch):  # exclamation:感叹语
    print(f" {exclamation}{epoch}")


f = show_progress("使用@函数方式：")
f2=show_progress("使用@函数方式 2 ：")

slow_calculation(f)

slow_calculation(f2)


class ProgressShowingCallback():
    def __init__(self, exclamation="Awesome"): self.exclamation = exclamation
    def __call__(self, epoch): print(f"{self.exclamation}! We've finished epoch {epoch}!")

cb = ProgressShowingCallback("使用class方式:")
slow_calculation(cb)

if __name__ == '__main__':
    print("----main()---------")
    a=f(3)
    print(f)
    print("  ---")
    print(a)
    pass


