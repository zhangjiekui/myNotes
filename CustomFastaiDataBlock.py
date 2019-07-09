# -*- coding: utf-8 -*-
# Author: Zhangjiekui
# Date: 2019-7-7 16:49
# torch.set_printoptions(linewidth=300)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# net=Net()
# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)
# net.to(device)

# sys.path.append('utils')  #for import module in utils
# from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax


# fastai version: 1.0.34 验证是正确的
from __future__ import absolute_import, division, print_function, unicode_literals
import pdb
import fastai
from fastai import *
from fastai.text import *
from fastai.tabular import *
from fastai.text.data import _join_texts
from pathlib import Path
import sys
import torch
import numpy as np

print(f'fastai version: {fastai.__version__}')
torch.cuda.set_device(1)
print(f'using GPU: {torch.cuda.current_device()}')

PATH=Path('data/yelp_dataset/')
PATH.mkdir(parents=True,exist_ok=True)
print(Path.cwd())

# sys.path.append('utils')  # for import module in utils

torch.set_printoptions(linewidth=300)


class MixedTabularLine(TabularLine):  # TabularLine is the sub_class of ItemBase: class TabularLine(ItemBase)
    def __init__(self, cat_cols_value_idxs:list, cont_cols_normalized_values, cat_cols_classes:OrderedDict, cat_cont_col_names:OptStrList, txt_idxs, txt_col_names, txt_string):
        '''
        代表DataFrame中的一行，由三部分组成
        <class 'list'>: [tensor([7170, 3012,7,311,2]), tensor([0.1650,1.5663,-0.3320,-0.3931]), array([2,4,545,12, ..., 215,51,792,615])]
        list[0]: 类别类型列对应的值
        list[1]：数值类型列对应的值
        list[2]：文本列对应的文本tokenize、numerize后对应的编码值
        :param cat_cols_value_idxs:list(int) ,such as tensor([7170,3012,7,311,2]),then by cat_classes['business_id'][7170] will be the business_id value
        :param cont_cols_normalized_values:list(int),such as tensor([ 0.1650,1.5663,-0.3320,-0.3931])
        :param cat_cols_classes:OrderedDict;such as OrderedDict('key={cat_cols_name}',array[value={列里所有的类别值}]
        :param cat_cont_col_names:list(str)，（类别类型列+数值类型列）列名list
        :param txt_idxs:list(int)，文本列对应的文本tokenize、numerize后对应的编码值
        :param txt_col_names:数值类型列列名，txt_col_names = ['text']
        :param txt_string:文本列中的tokenize处理后的文本值，例如'xxbos xxmaj although i had heard of xxmaj xxunk ,...'
        '''

        # using TabularLine's :super().__init__
        super().__init__(cat_cols_value_idxs, cont_cols_normalized_values, cat_cols_classes, cat_cont_col_names)

        # add the text bits
        self.txt_idxs=txt_idxs
        self.txt_col_names=txt_col_names
        self.text=txt_string

        # append numericalted text data to your input (represents your X values that are fed into your model)
        # self.data = [tensor(cat_cols_value_idxs), tensor(cont_cols_normalized_values), tensor(txt_idxs)]
        self.data+=[np.array(self.txt_idxs, dtype=np.int64)]
        self.obj=self.data

    def __str__(self):
        res=super().__str__()+f'Text:{self.text}'
        return res

class MixedTabularProcessor(TabularProcessor):
    tokenizer = Tokenizer(tok_func=BaseTokenizer, lang='en', pre_rules=[], post_rules=[], special_cases=[])

    # ItemList or ItemBase? class TabularProcessor(PreProcessor):def __init__(self, ds:ItemBase=None, procs=None):
    # todo ItemList or ItemBase? : ds:ItemList（MixedTabularList）

    def __init__(self,ds:ItemList=None,procs=None,tokenizer:Tokenizer=tokenizer,chunksize:int=10000,
                 vocab:Vocab=None,max_vocab:int=60000,min_freq:int=2):
        super().__init__(ds,procs)
        self.tokenizer, self.chunksize = ifnone(tokenizer, Tokenizer()), chunksize
        vocab=ifnone(vocab,ds.vocab if ds is not None else None)
        self.vocab, self.max_vocab, self.min_freq = vocab, max_vocab, min_freq

        #   # for testing process_one method
        # item = ds.get(0)  # df.Series
        # print(type(item))
        #
        # self.process(ds)
        # self.process_one(item)
        #
        # item1 = ds.get(0)  # df.Series
        # print(type(item1))
        #
        # self.process_one(item1)  #c错误，是供process方法调用的
        #
        #
        #
        # #     for testing process method
        # print("self.process(ds) -------------begin")
        # item = ds[0]
        # ds.text_cols
        #
        # print("self.process(ds) -------------end")



    # process a single item in a dataset
    # todo NOTE: THIS METHOD HAS NOT BEEN TESTED AT THIS POINT (WILL COVER IN A FUTURE ARTICLE)
    # process_one(item) 是供process方法调用的:
    def process_one(self, item): #item need to be type of df.Series
        # process tabular data (copied form tabular.data)
        df=pd.DataFrame([item,item])
        for proc in self.procs:
            proc(df,test=True) #todo
        # for proc in self.procs:
        #     proc(df, True)
        if len(self.cat_names)!=0:
            codes=np.stack([c.cat.codes.values for n,c in df[self.cat_names].items()],1).astype(np.int64)+1
        else:
            codes=[[]]

        if len(self.cont_names)!=0:
            conts=np.stack([c.astype('float32').values for n,c in df[self.cont_names].items()],1)
        else:
            conts=[[]]
        classes=None
        col_names=list(df[self.cat_names].columns.values)+list(df[self.cont_names].columns.values)
        # above:  process tabular data (copied form tabular.data)

        # below: process textual data (add the customed code lines below)
        if len(self.txt_col_names)!=0:
            txt=_join_texts(df[self.txt_col_names].values,(len(self.txt_col_names)>1))
            txt_toks=self.tokenizer._process_all_1(txt)[0]
            txt_ids=np.array(self.vocab.numericalize(txt_toks),dtype=np.int64)
        else:
            txt_toks,txt_ids=None,[[]]

        # return ItemBase
        return MixedTabularLine(codes[0],conts[0],classes,col_names,txt_ids,self.txt_col_names,txt_toks)

    # processes the entire dataset
    def process(self, ds):
        # pdb.set_trace()
        # process tabular data and then set "preprocessed=False" since we still have text data possibly
        super().process(ds)
        self.txt_col_names=ds.text_cols

        ds.preprocessed = False

        # process text data from column(s) containing text
        if len(ds.text_cols) != 0:
            texts = _join_texts(ds.xtra[ds.text_cols].values, (len(ds.text_cols) > 1))

            # tokenize (set = .text)
            tokens = []
            for i in progress_bar(range(0, len(ds), self.chunksize), leave=False):
                tokens += self.tokenizer.process_all(texts[i:i + self.chunksize])
            ds.text = tokens

            # set/build vocab
            if self.vocab is None: self.vocab = Vocab.create(ds.text, self.max_vocab, self.min_freq)
            ds.vocab = self.vocab
            ds.text_ids = [np.array(self.vocab.numericalize(toks), dtype=np.int64) for toks in ds.text]
        else:
            ds.text, ds.vocab, ds.text_ids = None, None, []

        ds.preprocessed = True


# each "ds" is of type LabelList(Dataset)
class MixedTabularDataBunch(DataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path: PathOrStr = '.', bs=64,
               pad_idx=1, pad_first=True, no_check: bool = False, **kwargs) -> DataBunch:
        # only thing we're doing here is setting the collate_fn = to our new "pad_collate" method above
        collate_fn = partial(mixed_tabular_pad_collate, pad_idx=pad_idx, pad_first=pad_first)

        return super().create(train_ds, valid_ds, test_ds, path=path, bs=bs, num_workers=1,
                              collate_fn=collate_fn, **kwargs)  #todo , no_check=no_check

        # return super().create(train_ds, valid_ds, test_ds, path=path, bs=bs, num_workers=1,
        #                       collate_fn=collate_fn, no_check=no_check, **kwargs)


# similar to the "fasta.text.data.pad_collate" except that it is designed to work with MixedTabularLine items,
# where the final thing in an item is the numericalized text ids.
# we need a collate function to ensure a square matrix with the text ids, which will be of variable length.
def mixed_tabular_pad_collate(samples: BatchSamples,
                              pad_idx: int = 1, pad_first: bool = True) -> Tuple[torch.LongTensor, torch.LongTensor]:
    "Function that collect samples and adds padding."

    samples = to_data(samples)
    max_len = max([len(s[0][-1]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx

    for i, s in enumerate(samples):
        if pad_first:
            res[i, -len(s[0][-1]):] = torch.LongTensor(s[0][-1])
        else:
            res[i, :len(s[0][-1]):] = torch.LongTensor(s[0][-1])

        # replace the text_ids array (the last thing in the inputs) with the padded tensor matrix
        s[0][-1] = res[i]

    # for the inputs, return a list containing 3 elements: a list of cats, a list of conts, and a list of text_ids
    return [x for x in zip(*[s[0] for s in samples])], tensor([s[1] for s in samples])


class MixedTabularList(TabularList):
    "A custom `ItemList` that merges tabular data along with textual data"

    _item_cls = MixedTabularLine
    _processor = MixedTabularProcessor
    _bunch = MixedTabularDataBunch

    def __init__(self, items: Iterator, cat_names: OptStrList = None, cont_names: OptStrList = None,
                 text_cols=None, vocab: Vocab = None, pad_idx: int = 1,
                 procs=None, **kwargs) -> 'MixedTabularList':
        # pdb.set_trace()
        super().__init__(items, cat_names, cont_names, procs, **kwargs)

        self.cols = [] if cat_names == None else cat_names.copy()
        if cont_names: self.cols += cont_names.copy()
        if txt_cols: self.cols += text_cols.copy()

        self.text_cols, self.vocab, self.pad_idx = text_cols, vocab, pad_idx

        # add any ItemList state into "copy_new" that needs to be copied each time "new()" is called;
        # your ItemList acts as a prototype for training, validation, and/or test ItemList instances that
        # are created via ItemList.new()
        self.copy_new += ['text_cols', 'vocab', 'pad_idx']

        self.preprocessed = False

    # defines how to construct an ItemBase from the data in the ItemList.items array
    def get(self, i):
        if not self.preprocessed:
            return self.xtra.iloc[i][self.cols] if hasattr(self, 'xtra') else self.items[i]

        codes = [] if self.codes is None else self.codes[i]
        conts = [] if self.conts is None else self.conts[i]
        text_ids = [] if self.text_ids is None else self.text_ids[i]
        text_string = None if self.text_ids is None else self.vocab.textify(self.text_ids[i])

        return self._item_cls(codes, conts, self.classes, self.col_names, text_ids, self.text_cols, text_string)

    # this is the method that is called in data.show_batch(), learn.predict() or learn.show_results()
    # to transform a pytorch tensor back in an ItemBase.
    # in a way, it does the opposite of calling ItemBase.data. It should take a tensor t and return
    # the same king of thing as the get method.
    def reconstruct(self, t: Tensor):
        return self._item_cls(t[0], t[1], self.classes, self.col_names,
                              t[2], self.text_cols, self.vocab.textify(t[2]))

    # tells fastai how to display a custom ItemBase when data.show_batch() is called
    def show_xys(self, xs, ys) -> None:
        "Show the `xs` (inputs) and `ys` (targets)."
        from IPython.display import display, HTML

        # show tabular
        display(HTML('TABULAR:<br>'))
        super().show_xys(xs, ys)

        # show text
        items = [['text_data', 'target']]
        for i, (x, y) in enumerate(zip(xs, ys)):
            res = []
            res += [' '.join([f'{tok}({self.vocab.stoi[tok]})'
                              for tok in x.text.split() if (not self.vocab.stoi[tok] == self.pad_idx)])]

            res += [str(y)]
            items.append(res)

        col_widths = [90, 1]

        display(HTML('TEXT:<br>'))
        display(HTML(text2html_table(items, (col_widths))))

    # tells fastai how to display a custom ItemBase when learn.show_results() is called
    def show_xyzs(self, xs, ys, zs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions)."
        from IPython.display import display, HTML

        # show tabular
        super().show_xyzs(xs, ys, zs)

        # show text
        items = [['text_data', 'target', 'prediction']]
        for i, (x, y, z) in enumerate(zip(xs, ys, zs)):
            res = []
            res += [' '.join([f'{tok}({self.vocab.stoi[tok]})'
                              for tok in x.text.split() if (not self.vocab.stoi[tok] == self.pad_idx)])]

            res += [str(y), str(z)]
            items.append(res)

        col_widths = [90, 1, 1]
        display(HTML('<br>' + text2html_table(items, (col_widths))))

    @classmethod
    def from_df(cls, df: DataFrame, cat_names: OptStrList = None, cont_names: OptStrList = None,
                text_cols=None, vocab=None, procs=None, xtra:DataFrame=None, **kwargs) -> 'ItemList':

        return cls(items=range(len(df)), cat_names=cat_names, cont_names=cont_names,
                   text_cols=text_cols, vocab=vocab, procs=procs, xtra=df, **kwargs)


if __name__ == '__main__':
    # data = (ImageList.from_folder(path)  # Where to find the data? -> in path and its subfolders
    #         .split_by_folder()  # How to split in train/valid? -> use the folders
    #         .label_from_folder()  # How to label? -> depending on the folder of the filenames
    #         .add_test_folder()  # Optionally add a test set (here default name is test)
    #         .transform(tfms, size=64)  # Data augmentation? -> use tfms with a size of 64
    #         .databunch())  # Finally? -> use the defaults for conversion to ImageDataBunch

    cat_cols = ['business_id', 'user_id', 'business_stars', 'business_postal_code', 'business_state']
    cont_cols = ['useful', 'user_average_stars', 'user_review_count', 'business_review_count']
    txt_cols = ['text']
    dep_var = ['stars']
    procs = [FillMissing, Categorify, Normalize]

    joined_df = pd.read_csv(PATH / 'joined_sample.csv', index_col=None)
    il = MixedTabularList.from_df(joined_df, cat_cols, cont_cols, txt_cols, vocab=None, procs=procs, path=PATH)

    print("il"*10)
    bil=il.get(0)
    print(bil)

    ils = il.random_split_by_pct(valid_pct=0.1, seed=42)

    print("-----------------------------------------")
    print("len(ils.train), len(ils.valid), ils.path")
    print(len(ils.train), len(ils.valid), ils.path)

    print("-----------------------------------------")
    ll = ils.label_from_df(dep_var)
    print("type(ll), type(ll.train), len(ll.lists)")
    print(type(ll), type(ll.train), len(ll.lists))

    print("----------------ll.train"+"----------------")
    print(ll.train)

    print("----------------all" + "----------------")
    print("ll.train.x[0]:",ll.train.x[0])

    print("ll.train.y[0]:", ll.train.y[0])

    print("ll.train.x.codes[0]:",ll.train.x.codes[0])

    print("ll.train.x.cat_names:",ll.train.x.cat_names)

    print("ll.train.x.text_ids[0]",ll.train.x.text_ids[0])

    # print(ll.train.x[0], ll.train.y[0], ll.train.x.codes[0], ll.train.x.cat_names, ll.train.x.text_ids[0])

    print("--------------------Length------------------------")

    print(len(ll.train.x.vocab.itos), len(ll.valid.x.vocab.itos))

    print("--------------------databunch------------------------")
    data_bunch = ll.databunch(bs=64)
    b = data_bunch.one_batch()
    print(len(b), len(b[0]), len(b[0][0]), len(b[0][1]), len(b[0][1]), b[1].shape)
    data_bunch.show_batch()


    # conts = np.stack([c.astype('float32').values for n, c in joined_df[cont_cols].items()], 1)
    #
    # codes = np.stack([c.cat.codes.values for n, c in joined_df[cont_cols].items()], 1).astype(np.int64) + 1

    # print(len(joined_df))
    # print(joined_df.head())
    # print(joined_df.describe().T)



    # TabularLine(ItemBase): def __init__(self, cats, conts, classes, names)
    # MixedTabularLine(TabularLine): def __init__(self, cats, conts, cat_classes, col_names, txt_ids, txt_cols, txt_string):

    # cat_cols = ['business_id', 'user_id', 'business_stars', 'business_postal_code', 'business_state']
    # cont_cols = ['useful', 'user_average_stars', 'user_review_count', 'business_review_count']
    # txt_ids=[0,1,2,3]
    # txt_string=['0s','1s','2s','3s']
    # txt_cols = ['text']
    # col_names=cat_cols+cont_cols+txt_cols
    # dep_var = ['stars']
    # mtl=MixedTabularLine(cat_cols,cont_cols,None,col_names,txt_ids,txt_cols,txt_string)

    pass