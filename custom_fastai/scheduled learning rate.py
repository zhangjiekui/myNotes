# https://github.com/fastai/course-v3/blob/master/nbs/dl2/05_anneal.ipynb

sched = combine_scheds([0.5, 0.5], [sched_cos(0.2, 1.), sched_cos(1., 0.1)]) 
sched1 = combine_scheds([0.3, 0.7], [sched_cos(0.6, 2.), sched_cos(2., 0.1)]) 
sched2 = combine_scheds([0.7, 0.3], [sched_cos(0.6, 2.), sched_cos(2., 0.1)]) 
sched3 = combine_scheds([0.2, 0.8], [sched_cos(0.6, 2.), sched_cos(2., 0.0)]) 
import numpy as np
lr=[]
lr1=[]
lr2=[]
lr3=[]
for pos in np.arange(0.,1.,0.01):
    lr.append(sched(pos))
    lr1.append(sched1(pos))
    lr2.append(sched2(pos))
    lr3.append(sched3(pos))

plt.plot(lr)
plt.plot(lr1)
plt.plot(lr2)
plt.plot(lr3)
print(lr[-1],lr1[-1],lr2[-1],lr3[-1])
