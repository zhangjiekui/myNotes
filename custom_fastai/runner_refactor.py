# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-8-10 10:52
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
import re
from exp.nb_03 import *

class DataBunch():
    def __init__(self,train_dl,valid_dl,c=None):
        self.train_dl,self.valid_dl=train_dl,valid_dl
        self.c=c
    @property
    def train_ds(self):return self.train_dl.dataset
    @property
    def valid_ds(self):return self.valid_dl.dataset

def get_model_and_opt(databunch,lr=0.5,nh=50):
    m=databunch.train_ds.x.shape[1]
    model=nn.Sequential(nn.Linear(m,nh),nn.ReLU(),nn.Linear(nh,c))
    return model,optim.SGD(model.parameters(),lr=lr)

class Learner():
    def __init__(self,data,model,opt,loss_func):
        self.data,self.model,self.opt,self.loss_func=data,model,opt,loss_func

# ==========================================================
#export
import re
from dataclasses import dataclass
@dataclass
class ModelTrainingHooks:
    begin_fit: str = 'begin_fit'
    begin_epoch: str = 'begin_epoch'
    begin_batch: str = 'begin_batch'
    begin_validate: str = 'begin_validate'
    # ==============================
    after_pred: str = 'after_pred'
    after_loss: str = 'after_loss'
    after_backward: str = 'after_backward'
    after_step: str = 'after_step'
    after_batch: str = 'after_batch'
    after_epoch: str = 'after_epoch'
    after_fit: str = 'after_fit'

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order = 0
    # 在其他地方可以使用：Callback.hooks.after_fit来调用，方便做提醒用
    hooks = ModelTrainingHooks()


    def set_runner(self, run): self.run=run
    # 如果Callback类本身没有k属性，就去Runner类里去找
    # 所以效果就是callback.attribute--->callback.attribute or runner.attribute
    def __getattr__(self, k): return getattr(self.run, k)
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

print("Callback.hooks.after_fit",Callback.hooks.after_fit)

class TrainEvalCallback(Callback):
    # callback.set_runner(self, run): self.run=run
    # 所以run在调用过callback.set_runner方法后，会是callback的属性之一，即self.run-->run
    def begin_fit(self):
        self.run.n_epochs = 0.
        print(f"self.n_epochs,self.run.n_epochs是等价的,{self.n_epochs},{self.run.n_epochs}")
        self.run.n_iter = 0

    def after_batch(self):
        if not self.run.in_train: return
        self.run.n_epochs += 1. / (self.run.iters)
        self.run.n_iter += 1


    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.run.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False














from typing import *

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


class Runner():
    def __init__(self, learn:Learner=None,cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            cbs.append(cb)
        self.stop, self.cbs = False, [TrainEvalCallback()] + cbs
        self.hooks=ModelTrainingHooks()

        if learn is None:
            print("初始化时的learn:Learner为None，请务必在fit()中要传入learn:Learner对象")
            self.learn=learn

    @property
    def opt(self):
        return self.learn.opt
    @property
    def model(self):
        return self.learn.model
    @property
    def loss_func(self):
        return self.learn.loss_func
    @property
    def data(self):
        return self.learn.data

    def one_batch(self, xb, yb):
        self.xb,self.yb = xb,yb
        # if self('begin_batch'): return
        if self(self.hooks.begin_batch): return
        self.pred = self.model(self.xb)
        # if self('after_pred'): return
        if self(self.hooks.after_pred): return
        self.loss = self.loss_func(self.pred, self.yb)
        print(
            f"      iters={self.iters},n_iter={self.n_iter},n_batch_iter={self.n_batch_iter},loss={self.loss},n_epochs={self.n_epochs}")

        # self.run.n_batch_iter=",self.run.n_batch_iter
        # if self('after_loss') or not self.in_train: return
        if self(self.hooks.after_loss) or not self.in_train: return
        self.loss.backward()
        # if self('after_backward'): return
        if self(self.hooks.after_backward): return
        self.opt.step()
        # if self('after_step'): return
        if self(self.hooks.after_step): return
        self.opt.zero_grad()

    def all_batches(self, dl):
        self.iters = len(dl)
        for xb,yb in dl:
            if self.stop: break
            self.one_batch(xb, yb)
            # self('after_batch')
            self(self.hooks.after_batch)
        self.stop=False

    def fit(self, epochs, learn):
        self.epochs = epochs
        if self.learn is None:
            self.learn=learn
        else:
            print(("初始化时已经提供了learn:Learner，现又在fit()中传入了非空的learn:Learner对象，请检查！"))
            raise Exception("重复的Learner对象参数！")

        try:
            for cb in self.cbs:
                cb.set_runner(self)
                setattr(self, cb.name, cb)  # todo 更改了位置，在此一次性注册所有的cb的属性
            # if self('begin_fit'): return
            if self(self.hooks.begin_fit): return
            for epoch in range(epochs):
                self.epoch = epoch
                # if not self('begin_epoch'): self.all_batches(self.data.train_dl)
                if not self(self.hooks.begin_epoch):
                    print(f"======training:   train=[{self.in_train}],epochs={self.epochs},epoch={self.epoch + 1},batch_size={self.data.train_dl.batch_size}")
                    self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    # if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                    if not self(self.hooks.begin_validate):
                        print(f"======validating: train=[{self.in_train}],epochs={self.epochs},epoch={self.epoch + 1},batch_size={self.data.valid_dl.batch_size}")
                        self.all_batches(self.data.valid_dl)
                # if self('after_epoch'): break
                if self(self.hooks.after_epoch): break

        finally:
            # self('after_fit')
            self(self.hooks.after_fit)
            self.learn = None

    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False

class TestCallback(Callback):
    _order = 1
    print("TestCallback(Callback)",Callback.hooks.after_fit)


    def __init__(self,batch_max_iter:int=0,all_max_iter=0):
        self.batch_max_iter=batch_max_iter
        self.all_max_iter = all_max_iter

    def begin_epoch(self):
        self.run.n_batch_iter=0

    def after_step(self):
        # print("self.train_eval.n_iter=",self.train_eval.n_iter)
        # print("self.run.n_iter=", self.run.n_iter)
        self.run.n_batch_iter+=1
        # print("self.run.n_batch_iter=",self.run.n_batch_iter)
        if self.run.n_iter>=self.all_max_iter>0:
            self.run.stop = True
            return True
        if self.run.n_batch_iter>=self.batch_max_iter>0:
            self.run.stop = True
            return True

class AvgStats():
    def __init__(self,metrics,in_train):
        self.metrics=listify(metrics)
        self.in_train=in_train
    def reset(self):
        self.tot_loss=0.
        self.count=0
        self.tot_mets=[0.]*len(self.metrics) #返回len(self.metrics)个【0.，0.,......】的列表
    @property
    def all_stats(self):
        l=[self.tot_loss.item()]
        # print("len of tot_loss.item()=",len(l))
        # print("len of self.tot_mets=",len(self.tot_mets))
        return l+self.tot_mets  #[loss]+[metrics_value]= [loss, metrics_value...]
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count:return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self,run):
        bn=run.xb.shape[0]
        self.tot_loss+=run.loss*bn
        self.count+=bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i]+=m(run.pred,run.yb)*bn

class AvgStatsCallback(Callback):
    def __init__(self,metrics):
        self.train_stats,self.valid_stats=AvgStats(metrics,True),AvgStats(metrics,False)
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
    def after_loss(self):
        stats=self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad():
            stats.accumulate(self.run)
    def after_epoch(self):
        print(f"   ***【epoch-{self.epoch} [Loss , Acc]】***: {self.train_stats},{self.valid_stats}")
        # print(self.valid_stats)

class Recorder(Callback):
    def begin_fit(self):
        self.lrs,self.losses=[],[]
    def after_batch(self):
        if not self.in_train:
            return
        self.lrs.append(self.opt.param_groups[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())
    def plot_lr(self):
        plt.plot(self.lrs)
    def plot_loss(self):
        plt.plot(self.losses)


class ParamScheduler(Callback):
    _order = 2

    def __init__(self, pname, sched_func):
        self.pname, self.sched_func = pname, sched_func

    def set_param(self):
        for pg in self.opt.param_groups:
            pg[self.pname] = self.sched_func(self.n_epochs / self.epochs)

    def begin_batch(self):
        if self.in_train: self.set_param()


from functools import partial
def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

#This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

# print(run.epoch)
if __name__ == '__main__':

    x_train, y_train, x_valid, y_valid = get_data()
    train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
    nh, bs = 50, 640
    c = y_train.max().item() + 1
    loss_func = F.cross_entropy

    databunch = DataBunch(*get_dls(train_ds, valid_ds, bs), c)
    model, opt = get_model_and_opt(databunch)

    learn = Learner(databunch, model, opt, loss_func)
    learner = learn

    tc = TestCallback(batch_max_iter=0,all_max_iter=0)
    print("TestCallback().name-->", tc.name)
    # rc = TrainEvalCallback()
    # run = Runner(cb_funcs=[TestCallback])

    stats = AvgStatsCallback([accuracy])
    run = Runner(cbs=[tc,stats])
    # run.train_eval
    print(f"run.cbs={run.cbs},len={len(run.cbs)}")

    # for xb, yb in run.data.train_dl:
    #     run.one_batch(xb, yb)
    print("##"*30)
    run.fit(10, learn)
    print("\n【Final [Loss , Acc]】:",stats.valid_stats.avg_stats)
    print("END")

    annealings = "NO LINEAR COS EXP".split()

    a = torch.arange(0, 100)
    p = torch.linspace(0.01, 1, 100)

    fns = [sched_no, sched_lin, sched_cos, sched_exp]
    # for fn, t in zip(fns, annealings):
    #     f = fn(2, 1e-2)
    #     plt.plot(a, [f(o) for o in p], label=t)
    # plt.legend();

    sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])
    # plt.plot(a, [sched(o) for o in p])

    cbfs = [Recorder,
            partial(AvgStatsCallback, accuracy),
            partial(ParamScheduler, 'lr', sched)]

    run = Runner(cbs=[tc,stats],cb_funcs=cbfs)
    # run.train_eval
    print(f"run.cbs={run.cbs},len={len(run.cbs)}")

    # for xb, yb in run.data.train_dl:
    #     run.one_batch(xb, yb)
    print("##"*30)
    run.fit(3, learn)
    print("\n【Final [Loss , Acc]】:",stats.valid_stats.avg_stats)
    print("END")
    # run.recorder.plot_lr()
    run.recorder.plot_loss()




    plt.show()





    pass


