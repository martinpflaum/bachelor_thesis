import torch
import torch.nn as nn
import torch.nn.functional as F

class Brainfeed(nn.Module):
    def __init__(self,down_dims = [4096,2048,1024],act_layer = nn.GELU,dropout = [0.1,0.1,0.025]):
        super(Brainfeed,self).__init__()
        #nn.SyncBatchNorm.convert_sync_batchnorm()
        pre = down_dims[0]
        layers = []
        for k in range(1,len(down_dims)):
            layers += [nn.BatchNorm1d(pre),nn.Linear(pre, down_dims[k]),act_layer(),nn.Dropout(dropout[k])]
            pre = down_dims[k]
            #layer += 
        final = [nn.BatchNorm1d(pre),nn.Linear(pre,512),nn.BatchNorm1d(512)]
        
        self.layers = nn.Sequential(*layers)
        self.final = nn.Sequential(*final)

        self.bias = torch.nn.Parameter(torch.zeros(()))#torch.nn.Parameter
        self.mul = torch.nn.Parameter(torch.ones(()))

    def forward(self,x):
        #if not isinstance(x, list):
        #    x = [x]
        
        #if len(x[0].shape)==1:
        #    x = [elem[None] for elem in x]
        #x = torch.cat(x,dim=0)
        x = self.layers(x)
        x = self.final(x)
        x = x*self.mul + self.bias
        return x

class Brainwrapper(nn.Module):
    def __init__(self,backbone,head):
        super(Brainwrapper,self).__init__()
        self.backbone = backbone
        self.head = head
    def forward(self,x):
        x = self.backbone(x)
        x = self.head(x)
        return x 

#from brainloading import BrainLoader,BRAIN_FILE_NAME_TRAIN
"""class Braindset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        brain_dataset_root = "D:/Datasets/BrainData"
        brain_data_file = BRAIN_FILE_NAME_TRAIN
        self.brain_loader = BrainLoader(brain_dataset_root,brain_data_file,mapToCupe=False,rel_n=4096)
        self.size = self.brain_loader.BRAIN_DATA_ARRAY.shape[0]
    def __len__(self):
        return self.size
    def __getitem__(self, index):
        alpha = torch.rand(3)
        alpha = nn.functional.softmax(alpha).reshape(3,1)
        x = []
        x += [self.brain_loader(index,0)[None]]
        x += [self.brain_loader(index,1)[None]]
        x += [self.brain_loader(index,2)[None]]
        x = torch.sum(alpha*torch.cat(x,dim=0),dim=0)
        #xb = torch.sum(alpha*torch.cat(x,dim=0),dim=0)
        return x,torch.rand(1)"""

#backbone = Backbone()
#backbone(torch.rand(3,4096)).shape
#%%

# %%