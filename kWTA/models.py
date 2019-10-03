import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision



class SparsifyBase(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super(SparsifyBase, self).__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()
        return hook
    
    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class Sparsify1D(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify1D, self).__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(self.sr*x.shape[1])
        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1,0)
        comp = (x>=topval).to(x)
        return comp*x



class Sparsify2D_chn(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_chn, self).__init__()
        self.sr = sparse_ratio

        self.preact = None
        self.act = None

    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        tmpx = x.view(x.shape[0], x.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:,:,-1]
        topval = topval.expand(x.shape[2], x.shape[3], x.shape[0], x.shape[1]).permute(2,3,0,1)
        comp = (x>=topval).to(x)
        return comp*x


class Sparsify2D_vol(SparsifyBase):
    '''cross channel sparsify'''
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_vol, self).__init__()
        self.sr = sparse_ratio


    def forward(self, x):
        size = x.shape[1]*x.shape[2]*x.shape[3]
        k = int(self.sr*size)

        tmpx = x.view(x.shape[0], -1)
        topval = tmpx.topk(k, dim=1)[0][:,-1]
        topval = topval.repeat(tmpx.shape[1], 1).permute(1,0).view_as(x)
        comp = (x>=topval).to(x)
        return comp*x


class Sparsify2D_abs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_abs, self).__init__()
        self.sr = sparse_ratio


    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2)[0][:,:,-1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2,3,0,1)
        comp = (absx>=topval).to(x)
        return comp*x

class Sparsify2D_invabs(SparsifyBase):
    def __init__(self, sparse_ratio=0.5):
        super(Sparsify2D_invabs, self).__init__()
        self.sr = sparse_ratio


    def forward(self, x):
        layer_size = x.shape[2]*x.shape[3]
        k = int(self.sr*layer_size)
        absx = torch.abs(x)
        tmpx = absx.view(absx.shape[0], absx.shape[1], -1)
        topval = tmpx.topk(k, dim=2, largest=False)[0][:,:,-1]
        topval = topval.expand(absx.shape[2], absx.shape[3], absx.shape[0], absx.shape[1]).permute(2,3,0,1)
        comp = (absx>=topval).to(x)
        return comp*x


sparse_func_dict = {
    'chn':Sparsify2D,  #top-k value per channel
    'abs':Sparsify2D_abs,  #top-k absolute value
    'invabs':Sparsify2D_invabs, #top-k minimal absolute value
    'vol':Sparsify2D_vol,  #cross channel top-k
}