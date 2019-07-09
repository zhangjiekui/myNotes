class PreProcessor():
    "Basic class for a processor that will be applied to items at the end of the data block API."
    def __init__(self, ds:Collection=None):  self.ref_ds = ds
    def process_one(self, item:Any):         return item
    def process(self, ds:Collection):        ds.items = array([self.process_one(item) for item in ds.items])
    
class TabularProcessor(PreProcessor):
    "Regroup the `procs` in one `PreProcessor`."
    def __init__(self, ds:ItemBase=None, procs=None):
        procs = ifnone(procs, ds.procs if ds is not None else None)
        self.procs = listify(procs)

    def process_one(self, item):
        df = pd.DataFrame([item,item])
        for proc in self.procs: proc(df, test=True)
        if len(self.cat_names) != 0:
            codes = np.stack([c.cat.codes.values for n,c in df[self.cat_names].items()], 1).astype(np.int64) + 1
        else: codes = [[]]
        if len(self.cont_names) != 0:
            conts = np.stack([c.astype('float32').values for n,c in df[self.cont_names].items()], 1)
        else: conts = [[]]
        classes = None
        col_names = list(df[self.cat_names].columns.values) + list(df[self.cont_names].columns.values)
        return TabularLine(codes[0], conts[0], classes, col_names)

    def process(self, ds):
        if ds.xtra is None:
            ds.classes,ds.cat_names,ds.cont_names = self.classes,self.cat_names,self.cont_names
            return
        for i,proc in enumerate(self.procs):
            if isinstance(proc, TabularProc): proc(ds.xtra, test=True)
            else:
                #cat and cont names may have been changed by transform (like Fill_NA)
                proc = proc(ds.cat_names, ds.cont_names)
                proc(ds.xtra)
                ds.cat_names,ds.cont_names = proc.cat_names,proc.cont_names
                self.procs[i] = proc
        self.cat_names,self.cont_names = ds.cat_names,ds.cont_names
        if len(ds.cat_names) != 0:
            ds.codes = np.stack([c.cat.codes.values for n,c in ds.xtra[ds.cat_names].items()], 1).astype(np.int64) + 1
            self.classes = ds.classes = OrderedDict({n:np.concatenate([['#na#'],c.cat.categories.values])
                                      for n,c in ds.xtra[ds.cat_names].items()})
            cat_cols = list(ds.xtra[ds.cat_names].columns.values)
        else: ds.codes,ds.classes,cat_cols = None,None,[]
        if len(ds.cont_names) != 0:
            ds.conts = np.stack([c.astype('float32').values for n,c in ds.xtra[ds.cont_names].items()], 1)
            cont_cols = list(ds.xtra[ds.cont_names].columns.values)
        else: ds.conts,cont_cols = None,[]
        ds.col_names = cat_cols + cont_cols
