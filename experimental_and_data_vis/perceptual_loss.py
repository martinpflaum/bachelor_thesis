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
#%%
import torch
import torch.nn as nn
import torchvision

class PerceptualLoss(nn.Module):
    def __init__(self,model,mean,std,img_loss_func=nn.MSELoss(),latent_loss_func=nn.MSELoss()):
        super().__init__()
        
        
        self.model = model


        self.latent_loss_func = latent_loss_func
        self.img_loss_func = img_loss_func
        
        self.transform = torchvision.transforms.Normalize(mean,std)
    def forward(self,pred,target):
        loss = 0
        loss += self.img_loss_func(pred,target)
        
        pred = self.transform(pred)
        target = self.transform(target)

        pred = self.model(pred)
        target = self.model(target)

        loss += self.latent_loss_func(pred,target)
        
        return loss

