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

