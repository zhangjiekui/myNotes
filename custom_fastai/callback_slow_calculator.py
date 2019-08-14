# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-8-14 14:35
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
from time import sleep
torch.Tensor.ndim = property(lambda x: len(x.shape))
torch.set_printoptions(linewidth=300, precision=4, sci_mode=False)

class SlowCalculator():
    def __init__(self,cb=None):
        self.cb=cb
        self.res=0
    def _callback(self, cb_name, *args):
        if self.cb and hasattr(self.cb,cb_name):
            cb=getattr(self.cb,cb_name)
            # return cb(*args)  #todo
            return cb(self,*args)  # 可以把（主动体）传递给cb的函数体内
    def calc(self):
        for i in range(5):
            self._callback('before_calc_No', i)
            self._callback('before_calc', i)
            self.res+=i*i
            if self._callback('after_calc', i, self.res):
                print(f'Stop early at epoch {i} , the result is {self.res}')

                break
        return self.res


class SlowCalculatorCallback():
    def __init__(self,stop_value=10):
        self.stop_value=stop_value

    # def before_calc(self, *args):
    def before_calc(self,slow_calculator:SlowCalculator,*args):
        print(f"计算前的结果. res={slow_calculator.res}")
        print(f"before_calc中传入的参数*args，i=:{args}")
    def after_calc(self,slow_calculator:SlowCalculator,*args):
        print(f"after_calc中传入的参数*args，i，res=:{args}")
        # 甚至可以在callback中更改主动体（slow_calculator）的值
        slow_calculator.res=slow_calculator.res*10
        print(f"在callback中更改主动体（slow_calculator）的值后:{slow_calculator.res}")




        if args[-1]>self.stop_value:
            return True

scb10=SlowCalculatorCallback()
scb25=SlowCalculatorCallback(25)

scalc=SlowCalculator(scb10)
scalc.calc()

scalc=SlowCalculator(scb25)
scalc.calc()

print("看看主动体的值是否发生变化：",scalc.res)


if __name__ == '__main__':
    pass


