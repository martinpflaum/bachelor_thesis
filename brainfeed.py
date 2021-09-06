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

        self.bias = torch.nn.Parameter(torch.zeros(()))#torch.zeros(())##torch.nn.Parameter
        self.mul = torch.nn.Parameter(torch.ones(()))#torch.ones(())#

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
