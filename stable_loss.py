"""
MIT License

Copyright (c) 2021 martinpflaum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn as nn

class StableLoss(nn.Module):
    def __init__(self,loss_func=nn.MSELoss(),alpha=0.):
        super(StableLoss,self).__init__()
    
        self.loss_func = loss_func
        self.latent_loss_func = nn.MSELoss()
        self.alpha = alpha
    def forward(self,xpred,target):
        if isinstance(xpred,tuple):
            pred = xpred[0]
            latent = xpred[1]
            latent_target = xpred[2].detach()
            
        else:
            return self.loss_func(xpred,target)
        B = target.shape[0]
        loss = 0
        
        latent_loss = self.latent_loss_func(latent,latent_target)*self.alpha
        loss += latent_loss
        loss += self.loss_func(pred,target)#/2
        
        return loss