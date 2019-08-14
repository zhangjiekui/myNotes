# 业务需要两个甚至多个参数，但函数定义只接受一个参数

# 函数定义只接受一个参数
from time import sleep
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        res += i*i
        sleep(1)
        if cb: cb(i)
    return res
    
 # 业务需要两个甚至多个参数
def show_progress(exclamation:str,epoch):  #exclamation:感叹语
    print(f" {exclamation}{epoch}")
    
    
 # 三种解决办法
def show_progress(exclamation:str,epoch):  #exclamation:感叹语
    print(f"{exclamation} {epoch} ")

def inner_show_progress(exclamation):
    def _inner_function(epoch):
        print(exclamation,epoch)
    return _inner_function

def lam_show_progress(exclamation):
    _inner_lambda= lambda epoch: print(f"{exclamation}:{epoch}")
    return _inner_lambda
       
from functools import partial
_partial_func=partial(show_progress,"使用partial函数：")  

_inner_func=inner_show_progress("使用内部函数：")
_inner_lamfunc=lam_show_progress("使用lambda函数：")
#-----------------------------------------------------

slow_calculation(_partial_func)
slow_calculation(_inner_func)
slow_calculation(_inner_lamfunc)

# 使用@函数方式：
def show_epochs(f):
    def _inner(epoch): return partial(f, epoch)
    return _inner

@show_epochs
def show_progress(exclamation:str,epoch):  #exclamation:感叹语
    print(f" {exclamation}{epoch}")
    
f=show_progress("使用@函数方式：")
f(3)
print(f,f(3))
slow_calculation(f)
