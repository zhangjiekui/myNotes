# https://github.com/fastai/course-v3/blob/master/nbs/dl2/07_batchnorm.ipynb

input = torch.randn(20, 5, 10, 10)
batch_norm=input.mean((0,2,3),keepdim=True)              # torch.Size([1, 5, 1, 1]), for classification
layer_norm=input.mean((1,2,3),keepdim=True)              # torch.Size([20, 1, 1, 1]), for RNN
instance_norm=input.mean((2,3),keepdim=True)             #torch.Size([20, 5, 1, 1]) , for style-transform
batch_norm.shape,layer_norm.shape,instance_norm.shape

class BatchNorm(nn.Module):
    def __init__(self,num_filers,mom=0.1,eps=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.mom,self.eps=mom,eps
        self.mults=nn.Parameter(torch.ones(num_filers,1,1))
        self.adds =nn.Parameter(torch.zeros(num_filers,1,1))
        self.register_buffer('vars',torch.ones(1,num_filers,1,1))
        self.register_buffer('means',torch.zeros(1,num_filers,1,1))
    def update_stats(self,x):
        m=x.mean((0,2,3),keepdim=True) # 取第二维的平均值，得到num_filers个平均数,结果的shape为(1,num_filers,1,1)
        v=x.var ((0,2,3),keepdim=True)
        self.means.lerp_(m,self.mom) # end, weight. start+w*(end-start)
        self.vars .lerp_(v,self.mom)
        return m,v
    
    def forward(self,x):
        if self.training:
            with torch.no_grad():
                m,v=self.update_stats(x)
        else:
            m,v=self.means,self.vars
        x=(x-m)/(v+self.eps).sqrt()
        return x*self.mults+self.adds
class LayerNorm(nn.Module):
    __constants__ = ['eps']
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mult = nn.Parameter(tensor(1.))
        self.add  = nn.Parameter(tensor(0.))

    def forward(self, x):
        m = x.mean((1,2,3), keepdim=True)
        v = x.var ((1,2,3), keepdim=True)
        x = (x-m) / ((v+self.eps).sqrt())
        return x*self.mult + self.add
