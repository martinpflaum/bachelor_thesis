#%%

from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import warnings
from torch.nn.functional import dropout, log_softmax
warnings.simplefilter("ignore", UserWarning)
import torchvision
import numpy as np
import PIL
import PIL.Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
import numpy as np

from data.seg_mask_loader import SegMaskLoader
from data.seg_mask_loader import seg_mask_to_color
from data.data_utils import get_train_val_test_split,open_image,\
    post_load_default,post_load_scene_depth,post_load_normal,\
    post_load_lightning,load_edge_tensor,load_pickle
from data.brainloading import BrainLoader, BRAIN_FILE_NAME_TRAIN, BRAIN_FILE_NAME_VAL
from data.list_transforms import LRandomHorizontalFlip,LRandomResizedCrop,LResize,LCompose,LCenterCrop

import random



def post_load_identity(img):
    return torchvision.transforms.functional.to_tensor(np.array(img))

class BrainDatasetSceneDepth(Dataset):
    def __init__(self,task,indicies,img_data_set_root,brain_dataset_root,brain_data_file,is_train,input_dim=30000):
        self.img_data_set_root = img_data_set_root
        self.is_train = is_train

        #self.multiplier = 6
        self.multiplier = 1
        
        if is_train:
            self.multiplier = 3
        

        self.size = len(indicies)*self.multiplier
        img_size = 64
        
        self.transform = LResize((img_size,img_size)) #Random resized crop has the right ration for our images
        self.brain_loader = BrainLoader(brain_dataset_root,brain_data_file,mapToCupe=False,rel_n=input_dim)

        self.indicies = indicies
        self.task = task
        
    def __len__(self):
        #print("get_len")
        return self.size
    def __getitem__(self, index):
        n = index % self.multiplier
        index = index //self.multiplier
        name = self.indicies[index]
    
        
        if self.is_train:
            x = self.brain_loader(index,n)
        else:
            x = []
            x += [self.brain_loader(index,0)[None]]
            x += [self.brain_loader(index,1)[None]]
            x += [self.brain_loader(index,2)[None]]
            x += [self.brain_loader(index,3)[None]]
            x += [self.brain_loader(index,4)[None]]
            x += [self.brain_loader(index,5)[None]]
            x += [self.brain_loader(index,6)[None]]
            x += [self.brain_loader(index,7)[None]]
            x += [self.brain_loader(index,8)[None]]


            x = torch.mean(torch.cat(x,dim=0),dim=0)


        y = None
        if self.task == "default":
            default = post_load_identity(open_image(name,"default",self.img_data_set_root))
            y = default
        if self.task == "scene_depth":
            scene_depth = post_load_scene_depth(open_image(name,"scene_depth",self.img_data_set_root))
            y = scene_depth
        if self.task == "world_normal":
            world_normal = post_load_normal(open_image(name,"world_normal",self.img_data_set_root),name,self.data_set_root)
            y = world_normal
        if self.task == "lightning":
            lightning = post_load_lightning(open_image(name,"lightning",self.img_data_set_root))
            y = lightning
        if self.task == "segmentation":
            raise NotImplementedError("segmentation is not supported for brain data")
            #segmentation = torchvision.transforms.functional.to_tensor(self.segMaskLoader(name))
        if self.task == "edges":
            edges = load_edge_tensor(name,self.img_data_set_root)
            y = edges
        if self.task == "reflectance":
            reflectance = post_load_default(open_image(name,"reflectance",self.img_data_set_root))
            y = reflectance
        
        y = self.transform([y])[0]

        return x,y


from fastai import *
from fastai.vision.all import *
from fastai.data.core import DataLoaders
import torch.optim as optim


from functools import partial
def apply_mod(m, f):
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b):
    if isinstance(m, (nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)

def set_absolute_grad(m, b):
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)

def split_layers(model):
    return L(model.pretrained,model.scratch).map(params)

import torch
from style_gan_2_model import normalize_2nd_moment,FullyConnectedLayer

from brainfeed import Brainfeed
class GenWrapper(nn.Module):
    def __init__(self,gan_file_name,input_dim=30000):
        super().__init__()
        self.backbone = Brainfeed(down_dims = [input_dim,2048,1024])
        self.bias = torch.tensor(128/255)
        self.mul = torch.tensor(127.5/255)
        generator = load_pickle(gan_file_name)
        generator = generator["G_ema"]
        self.gen = generator
        self.num_ws = generator.mapping.num_ws
        
    def forward(self,input,c = None,force_fp32=False):
        
        x = self.backbone(input)
        out = self.gen(x,c=None,force_fp32=force_fp32)
        out = out * self.mul + self.bias
        return out

#%%

def main(task):
    train,val,test = get_train_val_test_split("./data_splits/train_test_split.csv")
    train_ds = BrainDatasetSceneDepth(task,train,"D:/Datasets/BrainData/rendered_animations_final","D:/Datasets/BrainData",BRAIN_FILE_NAME_TRAIN,True)#True
    valid_ds = BrainDatasetSceneDepth(task,val,"D:/Datasets/BrainData/rendered_animations_final","D:/Datasets/BrainData",BRAIN_FILE_NAME_VAL,False)

    dls = DataLoaders.from_dsets(train_ds, valid_ds, batch_size = 16,num_workers = 0,device="cuda:0")
    model = GenWrapper(f"{task}_gan_2000.pkl")
    loss_func = nn.MSELoss()
    
    learn = Learner(dls,model,loss_func = loss_func)
    apply_mod(learn.model.gen, partial(set_absolute_grad, b = False))
    learn.fit(80)
    learn.save(f"{task}_learner")

if __name__ == '__main__':
    main("scene_depth")
    
# %%
task = "scene_depth"
train,val,test = get_train_val_test_split("./data_splits/train_test_split.csv")
train_ds = BrainDatasetSceneDepth(task,train,"D:/Datasets/BrainData/rendered_animations_final","D:/Datasets/BrainData",BRAIN_FILE_NAME_TRAIN,True)#True
valid_ds = BrainDatasetSceneDepth(task,val,"D:/Datasets/BrainData/rendered_animations_final","D:/Datasets/BrainData",BRAIN_FILE_NAME_VAL,False)
#%%
dls = DataLoaders.from_dsets(train_ds, valid_ds, batch_size = 8,num_workers = 0,device="cuda:0")
model = GenWrapper(f"{task}_gan_2000.pkl")
loss_func = nn.MSELoss()
learn = Learner(dls,model,loss_func = loss_func)
learn.load(f"{task}_learner")
learn.model = learn.model.to("cuda:0")
learn.model.eval()
#%%

import ntpath
import numpy as np 
import os
import pickle
import nilearn
import nibabel as nib
from nilearn import plotting
file_name = ntpath.basename(__file__)[:-3]
save_folder = "img_results"
from importance import noise_adjustment_global,eval_importance

from data.data_utils import save_pickle
from data.brainloading import load_reliability_mask,map2cube


def visualize_importance(importance,out_file):
    def saveasnii(brain_mask,nii_save_path,nii_data):
        img = nib.load(brain_mask)
        print(img.shape)
        nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
        nib.save(nii_img, nii_save_path)

    brain_data_root = "D:/Datasets/BrainData"
    subject = 3
    n_input = 30000
    rel_mask = load_reliability_mask(brain_data_root,n = n_input,subject = subject)
    out = np.zeros((100045))
    out[rel_mask] = importance

    nan_indices_path = os.path.join(brain_data_root + '/fMRI_data',f"sub{subject:02d}",'z_1_nan_indicesWB.npy')
    mask_path = os.path.join(brain_data_root + '/fMRI_data',f"sub{subject:02d}",'mask.nii')
                
    cube_data = map2cube(out,mask_path, nan_indices_path)
    saveasnii(mask_path,"./example.nii",cube_data)
    plotting.plot_glass_brain("./example.nii",colorbar=True,output_file=out_file)


im_noise_adjustment = noise_adjustment_global(valid_ds,learn.model).numpy()
visualize_importance(im_noise_adjustment,"noise_adj_importance_vis.png")

for n_keep in [30000,2900,2800,2700,2600,2500,2000,1500,1000,500,200,100]:
    eval_importance(learn.model,valid_ds,torch.tensor(im_noise_adjustment).to("cuda:0"),n_keep=n_keep)

print("valid_ds len",len(valid_ds))
from eval_metrics import run_all_metrics

run_all_metrics(save_folder,learn,valid_ds,file_name)

view = plotting.view_img_on_surf("./example.nii", threshold='80%', surf_mesh='fsaverage') 
view.save_as_html(file_name+".html") 
# %%
