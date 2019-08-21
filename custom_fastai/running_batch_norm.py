
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/07_batchnorm.ipynb

class RunningBatchNormUpdated(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('count', tensor(0.)) 
        
        self.register_buffer('factor', tensor(0.))# reviced         
        self.register_buffer('offset', tensor(0.))# reviced
        self.batch=0 #added

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1-self.mom)/math.sqrt(bs-1))
        self.sums.lerp_(s, mom1) #去掉self.mom1--》self.
        self.sqrs.lerp_(ss,mom1) #去掉self.mom1--》self.#去掉self.mom1--》self.
        self.count.lerp_(c,mom1) #去掉self.mom1--》self.
        self.batch += bs
        
        means = self.sums/self.count
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)

        self.factor=self.mults/(varns+self.eps).sqrt()
        self.offset=self.adds-means*self.factor
    
    def forward(self, x):
        if self.training: self.update_stats(x)
        return x*self.factor+self.offset




# 下面为对照版
# class RunningBatchNormUpdateV2(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('count', tensor(0.)) 
        
        self.register_buffer('factor', tensor(0.))# reviced         
        self.register_buffer('offset', tensor(0.))# reviced
#         self.register_buffer('batch', tensor(0.))# deleted
        self.batch=0 #added

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        #c = self.count.new_tensor(x.numel()/nc)
        c = s.new_tensor(x.numel()/nc)
        #mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)
        mom1 = s.new_tensor(1 - (1-self.mom)/math.sqrt(bs-1))
        #self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, mom1) #去掉self.mom1--》self.
        self.sqrs.lerp_(ss,mom1) #去掉self.mom1--》self.#去掉self.mom1--》self.
        self.count.lerp_(c,mom1) #去掉self.mom1--》self.
        #self.dbias = self.dbias*(1-self.mom1) + self.mom1
        self.batch += bs
        #self.step += 1
        
#         从forward方法中，调整位置后过来的代码
#         means = sums/c
        means = self.sums/self.count
#         vars = (sqrs/c).sub_(means*means)
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
            
#         x = (x-means).div_((vars.add_(self.eps)).sqrt())
#         return x.mul_(self.mults).add_(self.adds)
        self.factor=self.mults/(varns+self.eps).sqrt()
        self.offset=self.adds-means*self.factor
    
    def forward(self, x):
        if self.training: self.update_stats(x)
        return x*self.factor+self.offset
#         sums = self.sums
#         sqrs = self.sqrs
#         c = self.count
#         if self.step<100:
#             sums = sums / self.dbias
#             sqrs = sqrs / self.dbias
#             c    = c    / self.dbias
