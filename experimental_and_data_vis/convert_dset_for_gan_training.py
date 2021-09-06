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
import sys
import numpy
from PIL import Image
import numpy as np
from data.data_utils import open_image,get_rotation_matrix,rgb2gray,canny_edge
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
import torch

scale_down = transforms.Resize((64,64))
def transform_scene_depth(folder_name,task,dataset_root):
    img = open_image(folder_name,task,dataset_root)
    img = np.array(img)
    img = img[:,:,0] + img[:,:,1] * 256
    maximum_dist = 256 * 4
    img = img.astype(np.float32) / maximum_dist
    img = Image.fromarray((img * 256).astype(numpy.uint8))
    return img

def transform_identity(folder_name,task,dataset_root):
    img = open_image(folder_name,task,dataset_root)
    return img

def transform_normal(folder_name,task,dataset_root):
    img = open_image(folder_name,task,dataset_root)
    img = np.array(img)
    img = (img.astype(np.float32) / 255) * 2 - 1 #normalize
    
    rot_matrix = get_rotation_matrix(folder_name,dataset_root)
    img = (torch.tensor(img) @ torch.tensor(rot_matrix).float()).numpy()
    img = (img +1)/2
    img = img.astype(np.float32)
    img = np.clip(img * 255,0,255)
    
    img = Image.fromarray((img).astype(numpy.uint8), 'RGB')
    return img

def transform_edge_tensor(name,task,dataset_root):
    frame = 0
    img = np.array(open_image(name,"default",dataset_root,frame = frame)).astype(np.float32)
    img = rgb2gray(img)
    img = canny_edge(img)
    img = img.reshape(img.shape[0],img.shape[1],1)
    img = (img.astype(np.float32) / 255)
    img = np.clip(img * 255,0,255)[:,:,0]
    img = Image.fromarray((img).astype(numpy.uint8))
    return img
#%%
#F.hflip(img)
#"D:/ImageDatasetBig"
def convert_and_save(hflip,version,img_transform,task,dataset_root,gan_dset_root,folder_name):
    #"Anim_00001"

    img = img_transform(folder_name,task,dataset_root)
    img = scale_down(img)
    if hflip:
        img = transforms.functional.hflip(img)
    img.save(f'{gan_dset_root}/{version}_{folder_name}_{task}_f_0.png')

def all_folder_names_from_csv(file_name):
    df = pd.read_csv(file_name)
    df = df[["name"]]
    out = []
    for elem in np.array(df):
        out+=list(elem)
    return out
#"gan_dataset"
#"./data_splits/imagedata11520.csv"
def convert_all_folders(hflip,version,img_transform,task,gan_dset_root,dataset_root,all_names):
    print(len(all_names))
    for elem in all_names:
        if hflip:
            convert_and_save(True,f"{version}_1flip",img_transform,task,dataset_root,gan_dset_root,elem)
        convert_and_save(False,f"{version}_0flip",img_transform,task,dataset_root,gan_dset_root,elem)


from data.data_utils import get_train_val_test_split
img_dset_root = "D:/ImageDatasetBig"
img_split_file = "./data_splits/imagedata11520.csv"
#{"world_normal","scene_depth","lightning","reflectance","default"}


img_big_names = all_folder_names_from_csv(img_split_file)
train,val,test = get_train_val_test_split("./data_splits/train_test_split.csv")
brain_dset_root = "D:/Datasets/BrainData/rendered_animations_final"



#%%
convert_all_folders(False,"big_img_set",transform_scene_depth,"scene_depth","gan_dset_scene_depth",img_dset_root,img_big_names)
convert_all_folders(False,"brain_img_set",transform_scene_depth,"scene_depth","gan_dset_scene_depth",brain_dset_root,train)


# %%

convert_all_folders(False,"big_img_set",transform_identity,"default","gan_dset_default",img_dset_root,img_big_names)
convert_all_folders(False,"brain_img_set",transform_identity,"default","gan_dset_default",brain_dset_root,train)

#%%

folder_world_normal = "C:/Users/Martin/Desktop/gan_dset_world_normal"
convert_all_folders(False,"big_img_set",transform_normal,"world_normal",folder_world_normal,img_dset_root,img_big_names)
convert_all_folders(False,"brain_img_set",transform_normal,"world_normal",folder_world_normal,brain_dset_root,train)

#%%
gan_dset_edge = "C:/Users/Martin/Desktop/gan_dset_edge"
convert_all_folders(False,"big_img_set",transform_edge_tensor,"edge",gan_dset_edge,img_dset_root,img_big_names)
convert_all_folders(False,"brain_img_set",transform_edge_tensor,"edge",gan_dset_edge,brain_dset_root,train)


#%%

#%%

#normals?? --augpipe=custom_aug
#python train.py --outdir=./training-runs --data=C:\Users\Martin\Documents\GitHub\mti_for_brain_data\gan_dset_default --gpus=1 --kimg=2000 --metrics=none 



#python train.py --outdir=./training-runs --data=C:\Users\Martin\Documents\GitHub\mti_for_brain_data\gan_dset_scene_depth --gpus=1 --kimg=1000 --metrics=none --resume=scene_depth_gan_4000.pkl


#python train.py --outdir=./training-runs --data=C:\Users\Martin\Documents\GitHub\mti_for_brain_data\gan_dset_scene_depth --gpus=1 --kimg=1200 --metrics=none --resume=network-snapshot-000800.pkl