class ItemBase():
    "Base item type in the fastai library."
    def __init__(self, data:Any): self.data=self.obj=data
    def __repr__(self): return f'{self.__class__.__name__} {self}'
    def show(self, ax:plt.Axes, **kwargs): ax.set_title(str(self))
    def apply_tfms(self, tfms:Collection, **kwargs):
        if tfms: raise Exception('Not implemented')
        return self
        
class TabularLine(ItemBase):
    "Basic item for tabular data."
    def __init__(self, cats, conts, classes, names):
        self.cats,self.conts,self.classes,self.names = cats,conts,classes,names
        self.data = [tensor(cats), tensor(conts)]

    def __str__(self):
        res = ''
        for c, n in zip(self.cats, self.names[:len(self.cats)]):
            res += f"{n} {(self.classes[n][c])}; "
        for c,n in zip(self.conts, self.names[len(self.cats):]):
            res += f'{n} {c:.4f}; '
        return res
        
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
        self.txt_string=txt_string

        # append numericalted text data to your input (represents your X values that are fed into your model)
        # self.data = [tensor(cat_cols_value_idxs), tensor(cont_cols_normalized_values), tensor(txt_idxs)]
        self.data+=[np.array(self.txt_idxs, dtype=np.int64)]
        self.obj=self.data

    def __str__(self):
        res=super().__str__()+f'Text:{self.txt_string}'
        return res
