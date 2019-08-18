from typing import *

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

class ListContainer():
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res




if __name__ == '__main__':
    lc = ListContainer([1, 2, 3])
    print(lc[(0,1,2)])
    print(lc[0])
    print(lc[True, False, True])
    print(lc)

    lc2 = ListContainer([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    print(lc2[True, False, True, False, True, False, True, False, True, False, True, False])
    print(lc2)
    lc2[0]=100
    print(lc2)
    


    pass
