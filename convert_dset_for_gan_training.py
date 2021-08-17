#%%
import sys
import numpy
from PIL import Image
import numpy as np
from data.data_utils import open_image,get_rotation_matrix
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
    
    img = Image.fromarray((img * 256).astype(numpy.uint8))
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
#%%

#%%

#normals?? --augpipe=custom_aug
#python train.py --outdir=./training-runs --data=C:\Users\Martin\Documents\GitHub\mti_for_brain_data\gan_dset_default --gpus=1 --kimg=2000 --metrics=none 



#python train.py --outdir=./training-runs --data=C:\Users\Martin\Documents\GitHub\mti_for_brain_data\gan_dset_scene_depth --gpus=1 --kimg=1000 --metrics=none --resume=scene_depth_gan_4000.pkl


#python train.py --outdir=./training-runs --data=C:\Users\Martin\Documents\GitHub\mti_for_brain_data\gan_dset_scene_depth --gpus=1 --kimg=1200 --metrics=none --resume=network-snapshot-000800.pkl