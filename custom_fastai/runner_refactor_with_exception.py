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

    after_cancel_batch:str ='after_cancel_batch' #using in runner.one_batch()
    after_cancel_epoch: str = 'after_cancel_epoch'  # using in runner.all_batches()
    after_cancel_train: str = 'after_cancel_train'  # using in runner.fit()


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order = 0
    hooks = ModelTrainingHooks() # 在其他地方可以使用：Callback.hooks.after_fit来调用，方便做提醒用

    def set_runner(self, run):  #todo
        self.run=run

    # 如果Callback类本身没有k属性，就去Runner类里去找
    # 所以效果就是callback.attribute--->callback.attribute or runner.attribute
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name:str):
        cbf=getattr(self,cb_name,None)
        if cbf and cbf():return True  # True 意味着要退出或停止
        return False # False 意味着没有问题发生，继续执行


# print("Callback.hooks.after_fit",Callback.hooks.after_fit)

class TrainEvalCallback(Callback):
    # callback.set_runner(self, run): self.run=run
    # Runner初始化函数中，所有callbacks循环调用过callback.set_runner方法后，会是callback的属性之一，即self.run-->run
    def begin_fit(self):
        self.run.p_epochs = 0. # 已运行的epochs比例
        # callback.attribute - -->callback.attribute or runner.attribute
        # print(f"self.p_epochs,self.run.p_epochs是等价的,{self.p_epochs},{self.run.p_epochs}")
        self.run.n_iter = 0

    def after_batch(self):
        if not self.run.in_train: return
        self.run.p_epochs += 1. / (self.run.iters) # run.iters = len(dl)  dataloader中的batches数量
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.p_epochs = self.epoch # 每个epoch开始，将p_epochs=epoch，然后该epoch中的每个batch中改变after_batch(self)： self.run.p_epochs += 1. / (self.run.iters) #run.iters
        self.run.model.train()
        self.run.in_train = True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train = False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


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
        # self.hooks=ModelTrainingHooks()
        self.hooks = Callback.hooks

        # 从fit()方法中挪到这里来，或许更好 ，可以保证从一开始就可以调用
        for cb in self.cbs:
            cb.set_runner(self)
            # setattr(x, 'foobar', 123) --> x.foobar = 123
            # setattr(self, cb.name, cb) --> run.callback_name=callback_name()
            setattr(self, cb.name, cb)  # todo 更改了位置，在此一次性注册所有的cb的属性

        self.learn = learn
        if learn is None: # 只是为了起到提醒作用！
            print("初始化时的learn:Learner为None，请务必在fit()中要传入learn:Learner对象")


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
        try:
            self.xb,self.yb = xb,yb
            # # if self('begin_batch'): return
            # if self(self.hooks.begin_batch): return
            self(self.hooks.begin_batch)

            self.pred = self.model(self.xb)
            # # if self('after_pred'): return
            # if self(self.hooks.after_pred): return
            self(self.hooks.after_pred)

            self.loss = self.loss_func(self.pred, self.yb)
            message=f"      iters={self.iters},n_iter={self.n_iter},n_batch_iter={self.n_batch_iter},loss={self.loss:.4f},p_epochs={self.p_epochs:.2f}"
            print(message)

            # # self.run.n_batch_iter=",self.run.n_batch_iter
            # # if self('after_loss') or not self.in_train: return
            # if self(self.hooks.after_loss) or not self.in_train: return
            self(self.hooks.after_loss)
            if not self.in_train: return

            self.loss.backward()
            # # if self('after_backward'): return
            # if self(self.hooks.after_backward): return
            self(self.hooks.after_backward)

            self.opt.step()

            # # if self('after_step'): return
            # if self(self.hooks.after_step): return
            self(self.hooks.after_step)

            self.opt.zero_grad()
        except CancelBatchException:self(self.hooks.after_cancel_batch)
        finally:self(self.hooks.after_batch)

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,yb in dl:
                self.one_batch(xb,yb)
        except CancelEpochException:self(self.hooks.after_cancel_epoch)
        finally:
            pass
            # print(f"   ***In Runner.all_batches(), finally block doing nothing now !  Rewrite in future if needed.")
        # for xb,yb in dl:
        #     if self.stop: break
        #     self.one_batch(xb, yb)
        #     # self('after_batch')
        #     self(self.hooks.after_batch)
        # self.stop=False

    def fit(self, epochs, learn):
        self.epochs = epochs
        self.loss=tensor(0.)
        if self.learn is None:  # 如果__init__时初始化的self.learn为None
            self.learn=learn
        else:
            print(("初始化时已经提供了learn:Learner，现又在fit()中传入了非空的learn:Learner对象，请检查！"))
            raise Exception(f"重复的Learner对象参数！已有self.learn={self.learn}，又传入了learn={learn}!")

        try:
            # # for cb in self.cbs:
            # #     cb.set_runner(self)
            # #     # setattr(x, 'foobar', 123) --> x.foobar = 123
            # #     # setattr(self, cb.name, cb) --> run.callback_name=callback_name()
            # #     setattr(self, cb.name, cb)  # todo 更改了位置，在此一次性注册所有的cb的属性
            #
            # # 所有的callback对象，通过上面的语句，可以使用self.callback_name or run.callback_name来调用
            # # 所有的callback对象中的方法，通过下面的def __call__(self, cb_name)方法，如self("begin_fit")来依次链式调用
            #
            # # 另外，在CallBack类中，def __getattr__(self, k): return getattr(self.run, k)，会把查找获取属性代理给run
            # # 因此，callback对象就可以直接调用run中的属性和方法，如cb.fit（）, cb.data等
            #
            # # if self('begin_fit'): return
            # if self(self.hooks.begin_fit): return

            self(self.hooks.begin_fit)

            for epoch in range(self.epochs):
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
                # # if self('after_epoch'): break
                # if self(self.hooks.after_epoch): break
                self(self.hooks.after_epoch)

        except CancelTrainException:self(self.hooks.after_cancel_train)

        finally:
            # self('after_fit')
            self(self.hooks.after_fit)
            self.learn = None

    def __call__(self, cb_name):
        # for cb in sorted(self.cbs, key=lambda x: x._order):
        #     cbf = getattr(cb, cb_name, None)
        #     if cbf and cbf(): return True
        # return False
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order):
            res = cb(cb_name) or res
        return res

class TestCallback(Callback):
    _order = 1
    print("TestCallback(Callback)",Callback.hooks.after_fit)


    def __init__(self,batch_max_iter:int=0,all_max_iter=0,continue_next_batch=True):
        self.batch_max_iter=batch_max_iter
        self.all_max_iter = all_max_iter
        self.continue_next_batch=continue_next_batch

    def begin_epoch(self):
        self.run.n_batch_iter=0

    def after_step(self):
        # print("self.train_eval.n_iter=",self.train_eval.n_iter)
        # print("self.run.n_iter=", self.run.n_iter)
        self.run.n_batch_iter+=1
        # print("self.run.n_batch_iter=",self.run.n_batch_iter)

        # if self.run.n_iter>=self.all_max_iter>0:
        #     self.run.stop = True
        #     return True
        # if self.run.n_batch_iter>=self.batch_max_iter>0:
        #     self.run.stop = True
        #     return True

        # cancel_flag = self.run.n_iter>=self.all_max_iter>0 or self.run.n_batch_iter>=self.batch_max_iter>0
        # if cancel_flag:
        #     print(f"self.run.n_iter={self.run.n_iter},self.all_max_iter={self.all_max_iter}")
        #     print(f"self.run.n_batch_iter={self.run.n_batch_iter},self.batch_max_iter={self.batch_max_iter}")
        #     # raise CancelTrainException()
        #
        #     if self.continue_next_batch:
        #         self(self.hooks.after_batch)
        #     else:
        #         raise CancelTrainException()

        if self.run.n_iter>=self.all_max_iter>0:
            print(f"   ***self.run.n_iter={self.run.n_iter},self.all_max_iter={self.all_max_iter}")
            raise CancelTrainException()
        if self.run.n_batch_iter>=self.batch_max_iter>0:
            if self.continue_next_batch:
                # self.run(self.hooks.begin_validate)
                print(f"   ***self.run.n_batch_iter={self.run.n_batch_iter},self.batch_max_iter={self.batch_max_iter}")
                raise CancelEpochException()


            else:
                print(f"   ***self.run.n_batch_iter={self.run.n_batch_iter},self.batch_max_iter={self.batch_max_iter}")
                raise CancelTrainException()


class AvgStats():
    def __init__(self,metrics,in_train):
        self.metrics=listify(metrics)
        self.in_train=in_train
    def reset(self):
        self.tot_loss=tensor(0.)
        self.count=0
        self.tot_mets=[tensor(0.)]*len(self.metrics) #返回len(self.metrics)个【0.，0.,......】的列表
    @property
    def all_stats(self):
        l=[self.tot_loss.item()]
        # print("len of tot_loss.item()=",len(l))
        # print("len of self.tot_mets=",len(self.tot_mets))
        return l+self.tot_mets  #[loss]+[metrics_value]= [loss, metrics_value...]
    @property
    def avg_stats(self):
        try:
            return [o/self.count for o in self.all_stats]
        except:
            return "因为前面取消，所以此处无值！"

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
        # self.lrs,self.losses=[],[]
        self.lrs = [[] for _ in self.opt.param_groups]
        self.losses = []

    def after_batch(self):
        if not self.in_train:
            return
        # self.lrs.append(self.opt.param_groups[-1]['lr'])
        for pg, lr in zip(self.opt.param_groups, self.lrs): lr.append(pg['lr'])
        self.losses.append(self.loss.detach().cpu())
    # def plot_lr(self):  plt.plot(self.lrs)
    def plot_lr(self, pgid=-1):
        plt.plot(self.lrs[pgid])
    # def plot_loss(self):        plt.plot(self.losses)
    def plot_loss(self, skip_last=0):
        plt.plot(self.losses[:len(self.losses) - skip_last])
    # 下面方法是新增的
    def plot(self, skip_last=0, pgid=-1):
        losses = [o.item() for o in self.losses]
        lrs = self.lrs[pgid]
        n = len(losses) - skip_last
        plt.xscale('log')
        plt.plot(lrs[:n], losses[:n])


class ParamScheduler(Callback):
    _order = 2

    # def __init__(self, pname, sched_func):        self.pname, self.sched_func = pname, sched_func
    def __init__(self, pname, sched_funcs):
        self.pname, self.sched_funcs = pname, sched_funcs
    # 下面方法是新增的
    def begin_fit(self):
        if not isinstance(self.sched_funcs, (list,tuple)):
            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)

    def set_param(self):
        assert len(self.opt.param_groups) == len(self.sched_funcs) #此行是新增的
        # for pg in self.opt.param_groups:
        #     pg[self.pname] = self.sched_func(self.p_epochs / self.epochs)
        for pg,f in zip(self.opt.param_groups,self.sched_funcs):
            # pg[self.pname] = f(self.n_epochs/self.epochs)
            pg[self.pname] = f(self.p_epochs / self.epochs)

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

def plot_lr_curve():
    max_iters, min_lr, max_lr = 100, 1e-6, 10
    lrs = []
    for n_iter in range(max_iters):
        pos = n_iter / max_iters
        #     print( pos)
        lr = min_lr * (max_lr / min_lr) ** pos
        #     print( f"{lr:.6f}")
        lrs.append(lr)
    import matplotlib.pyplot as plt
    plt.plot(lrs)
    plt.show()

class LR_Find(Callback):
    _order = 1
    def __init__(self, max_iter =100 , min_lr = 1e-6 , max_lr = 10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9
    def begin_batch(self):
        if not self.in_train : return
        pos = self.n_iter/self.max_iter
        lr=self.min_lr * (self.max_lr/self.min_lr)**pos # lr曲线可以看上面的plot_lr_curve（）
        for pg in self.opt.param_groups:
            pg['lr']=lr
    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss<self.best_loss:
            self.best_loss=self.loss

def plot_anneling_curve():

    annealings = "NO LINEAR COS EXP".split()
    a = torch.arange(0, 100)
    p = torch.linspace(0.01, 1, 100)
    fns = [sched_no, sched_lin, sched_cos, sched_exp]
    fig = plt.figure()
    for fn, t in zip(fns, annealings):
        f = fn(2, 1e-2)
        plt.plot(a, [f(o) for o in p], label=t)
    plt.legend()
    fig.show()
    fig = plt.figure()

    global sched
    sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])
    plt.plot(a, [sched(o) for o in p])
    fig.show()


def test1():
    global c
    x_train, y_train, x_valid, y_valid = get_data()
    train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
    nh, bs = 50, 640
    c = y_train.max().item() + 1
    loss_func = F.cross_entropy
    databunch = DataBunch(*get_dls(train_ds, valid_ds, bs), c)
    model, opt = get_model_and_opt(databunch)
    learn = Learner(databunch, model, opt, loss_func)
    learner = learn



    # print("TestCallback().name-->", tc.name)
    # rc = TrainEvalCallback()
    # run = Runner(cb_funcs=[TestCallback])

    tc = TestCallback(batch_max_iter=0, all_max_iter=0)
    stats = AvgStatsCallback([accuracy])

    sched = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])

    cbfs_sched = [Recorder,
            partial(AvgStatsCallback, accuracy),
            partial(ParamScheduler, 'lr', sched)]
    run = Runner(cbs=[tc, stats], cb_funcs=cbfs_sched)

    cbfs_lr_find = [Recorder,
            # partial(AvgStatsCallback, accuracy),
            LR_Find]

    run = Runner(cbs=[tc, stats], cb_funcs=cbfs_lr_find)
    # run.train_eval
    # print(f"run.cbs={run.cbs.__class__.__name__},len={len(run.cbs)}")
    print(f"run.cbs={', '.join([cb.name for cb in run.cbs])}, len={len(run.cbs)}")
    print("\n---------------------------" * 2)
    run.fit(2, learn)

    fig1 = plt.figure("LR")
    plt.title("LR")
    run.recorder.plot_lr()
    fig1.show()

    fig2 = plt.figure()
    plt.title("LOSS")
    run.recorder.plot_loss()
    fig2.show()

    fig3 = plt.figure()
    plt.title("LR_Find_Loss")
    run.recorder.plot(skip_last=5)
    fig3.show()

    plt.show()


# print(run.epoch)
if __name__ == '__main__':

    # plot_lr_curve()
    # plot_anneling_curve()
    test1()

    pass


