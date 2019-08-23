# -*- coding: utf-8 -*-

# Author: HP/ZhangJieKui
# Date: 2019-8-23 09:00
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
import PIL,os,mimetypes

from exp.nb_07a import *

torch.Tensor.ndim = property(lambda x: len(x.shape))
torch.set_printoptions(linewidth=300, precision=4, sci_mode=False)

def setify(o):return o if isinstance(o,set) else set(listify(o))

def _get_files(path, fs, extensions=None):
    path = Path(path)
    if extensions is None:
        extensions = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))
    res = [path / f for f in fs if
           not f.startswith('.') and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res


def get_files(path, extensions=None, recurse=False, include=None):
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (p, d, f) in enumerate(os.walk(path)):  # returns (dirpath, dirnames, filenames)
            if include is not None and i == 0:
                d[:] = [o for o in d if o in include]
            else:
                d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, f, extensions)

def compose(x, funcs, *args, order_key='_order', **kwargs):
    from inspect import isfunction
    key = lambda o: getattr(o, order_key, 0)
    print(
        f"组合方法执行顺序：{['Function_' + item.__name__ if isfunction(item) else 'ClasObj_' + item.__class__.__name__ for item in sorted(listify(funcs), key=key)]}")
    print('------------------------------------------\n')
    for f in sorted(listify(funcs), key=key):
        x = f(x, *args, **kwargs)
    return x


class ListContainer():
    def __init__(self, items):
        self.items = listify(items)
        print("__init__")

    def __getitem__(self, idx):
        try:
            return self.items[idx]
        except TypeError:
            if isinstance(idx[0], bool):
                assert len(idx) == len(self)  # bool mask
                return [o for m, o in zip(idx, self.items) if m]
            return [self.items[i] for i in idx]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, i, o):
        self.items[i] = o

    def __delitem__(self, i):
        del (self.items[i])

    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n  Items: {self.items[:10]}'
        if len(self) > 10: res = res[:-1] + '...]'
        return res


class ItemList(ListContainer):
    def __init__(self, items, path='.', tfms=None):
        super().__init__(items)
        self.path, self.tfms = Path(path), tfms

    def __repr__(self):
        from inspect import isfunction
        tfms_repr = "None" if self.tfms is None else [
            'Function_' + item.__name__ if isfunction(item) else 'ClasObj_' + item.__class__.__name__ for item in
            listify(self.tfms)]
        return f"{super().__repr__()} \n  Path: {self.path} \n  Tfms: {tfms_repr}"

    # def __new__(self,items, path='.', tfms=None):是不一样的
    def new(self, items, cls=None):
        print("new")
        if cls is None: cls = self.__class__
        return cls(items, path=self.path, tfms=self.tfms)

    def get(self, any_object):
        return any_object  # todo 似乎是个bug

    def _get(self, i):
        # self.get(i)实际是获取到待转换的x对象；例如，获取一张图片，并交给self.tfms转换，
        # 该方法供__getitem__调用
        return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, list): return [self._get(o) for o in res]
        return self._get(res)


class ImageList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None: extensions = image_extensions
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    # image_list[i],首先调用父类的__getitem__，得到一个item（此处是一个图片地址），然后PIL.Image.open(file_name)打开这个图片
    # self.get(i)实际是获取到待转换的x对象；例如，获取一张图片，并交给self.tfms转换，
    # 该方法供__getitem__调用
    def get(self, file_name):
        return PIL.Image.open(file_name)

def grandparent_splitter(fn,valid_name='valid',train_name='train'):
    gp=fn.parent.parent.name
    is_valid=None
    if gp==valid_name:
        is_valid=True
    elif gp==train_name:
        is_valid=False
#     print(is_valid)
    return is_valid

# 返回的是ds.items 的列表
def split_by_func(ds,func):
    items=ds.items
    mask=[func(o) for o in items]
    train=[o for o, m in zip(items,mask) if m==False]
    valid=[o for o, m in zip(items,mask) if m==True]
    return train,valid

class SplitData():
    def __init__(self,train,valid):self.train,self.valid=train,valid
    def __getattr__(self,k):return(self.train,k)
    @classmethod
    def split_by_func(cls,item_list:ImageList,split_func):
        # train,valid=split_by_func(item_list,split_func)# split_by_func(item_list,split_func): 返回的是train,valid对应的 items
        # return item_list.new(train),item_list.new(valid)

        # split_by_func(item_list,split_func): 返回的是train,valid对应的 items
        lists=map(item_list.new, split_by_func(item_list,split_func))
        return cls(*lists)
    def __repr__(self):
        return f'ClassName({self.__class__.__name__})\nTrain:---\n{self.train}\nValid---:\n{self.valid}\n'

splitter=partial(grandparent_splitter,valid_name='val')

from collections import OrderedDict


def uniquify(x, sort=False):
    # set(list) 是会直接排序的！  s=[1,1,14,5,5,6,2,3]
    # uniquify(s)  [1, 14, 5, 6, 2, 3]
    # set(s)      {1, 2, 3, 5, 6, 14}
    res = list(OrderedDict.fromkeys(x).keys())
    if sort: res.sort()
    return res


class Processor():
    def process(self, items): return items
    def __call__(self, items):
        return self.process(items)

class CategoryProcessor(Processor):
    def __init__(self): self.vocab = None
    # The vocab is defined on the first use.
    def process(self, items):
        if self.vocab is None:
            self.vocab = uniquify(items)
            self.otoi = {v: k for k, v in enumerate(self.vocab)}
        return [self.procl(o) for o in items]
    # 在compose中，必须是func（x）
    # def __call__(self, items):  在父类中实现的
    #     return self.process(items)



    def procl(self, item):
        return self.otoi[item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deprocl(idx) for idx in idxs]

    def deprocl(self, idx):
        return self.vocab[idx]


def parent_labeler(fn: Path):
    #     print(type(fn))
    return fn.parent.name

#[f(o) for o in ds.items] 根据图片文件的路径，得到类别Label
#然后生成ItemList实例
def _label_by_func(ds, f, cls=ItemList): return cls([f(o) for o in ds.items], path=ds.path)


class LabeledData():
    def process(self, il, proc): return il.new(compose(il.items, proc))

    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x = x if proc_x is None else self.process(x, proc_x)
        self.y=  y if proc_y is None else self.process(y, proc_y)
        self.proc_x, self.proc_y = proc_x, proc_y

    def __repr__(self):
        return f'ClassName({self.__class__.__name__})\nXs:---\n{self.x}\nYs---:\n{self.y}\n'

    def __getitem__(self, idx): return self.x[idx], self.y[idx]

    def __len__(self): return len(self.x)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx, torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.deproc1(item) if isint else proc.deprocess(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        # x=il y=_label_by_func(x, f)
        y= _label_by_func(il, f)
        return cls(il, y, proc_x=proc_x, proc_y=proc_y)
        # return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)


def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train, valid)

if __name__ == '__main__':
    path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
    print(path)
    image_extensions = set(k for k, v in mimetypes.types_map.items() if v.startswith('image/'))
    iml = ImageList.from_files(path)
    img=iml[2]
    plt.imshow(img)
    plt.show()

    splited_data=SplitData.split_by_func(iml,splitter)
    # print(splited_data)

    print(len(splited_data.train),len(splited_data.valid))

    ll = label_by_func(splited_data, parent_labeler, proc_y=CategoryProcessor())
    print("------------------------")
    print(ll)

    pass


