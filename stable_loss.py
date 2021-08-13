
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