#%%
from fastai.vision.core import size
from scipy.ndimage.measurements import label
from data.seg_mask_loader import SegMaskLoader,seg_mask_to_color
import numpy as np
import matplotlib.pyplot as plt
from data.data_utils import get_train_val_test_split,open_image,\
    post_load_default,post_load_scene_depth,post_load_normal,\
    post_load_lightning,load_edge_tensor,load_pickle,get_rotation_matrix
import torch

#%%
dataset_root = "D:/ImageDatasetBig"
seg_loader = SegMaskLoader("./data/version2.1.pickle",dataset_root)
# %%

def get_seg_img(index):
    folder_name = f"Anim_{index+1:05}"
    img = seg_loader(folder_name)
    img = seg_mask_to_color(img)/255
    img = np.clip(img,0,1)
    return img

# %%

def get_normal(index):
    folder_name = f"Anim_{index+1:05}"
    img = open_image(folder_name,"world_normal",dataset_root)
    img = np.array(img)
    img = (img.astype(np.float32) / 255) * 2 - 1 #normalize

    rot_matrix = get_rotation_matrix(folder_name,dataset_root)
    img = (torch.tensor(img) @ torch.tensor(rot_matrix).float()).numpy()
    img = (img + 1)/2
    img = np.clip(img,0,1)
    return img

def get_edges(index):
    folder_name = f"Anim_{index+1:05}"
    img = load_edge_tensor(folder_name,dataset_root,frame = 0)[0].numpy()
    return img

def get_reflectance(index):
    folder_name = f"Anim_{index+1:05}"
    img = open_image(folder_name,"reflectance",dataset_root)
    img = np.array(img)/255
    img = np.clip(img,0,1)
    return img


def get_lightning(index):
    folder_name = f"Anim_{index+1:05}"
    img = open_image(folder_name,"lightning",dataset_root)
    img = np.array(img)/255
    img = np.clip(img,0,1)
    return img


def get_default(index):
    folder_name = f"Anim_{index+1:05}"
    img = open_image(folder_name,"default",dataset_root)
    img = np.array(img)/255
    img = np.clip(img,0,1)
    return img


# %%
index = 0
# %%
img = get_edges(index)
# %%
img = get_reflectance(index)
#%%

plt.imshow(get_normal(index))
# %%

import tempfile
import pandas as pd

def load_skeleton_data(index):
    folder_name = f"Anim_{index+1:05}"
    df = pd.read_csv(dataset_root+"/"+f"{folder_name}/skeleton_positions_screen_f_0.csv")
    skeleton_pos = df[["screen_pos_x","screen_pos_y"]].to_numpy()
    return skeleton_pos,np.array(df[["bone_name"]])

#%%
import copy

import PIL
def draw_points(skeleton_points, image):
    ys,xs,_ = image.shape
    xh = xs//2
    yh = ys//2
    
    image = copy.deepcopy(image)
    l = 4
    skeleton_points = skeleton_points#[0][None]
    for x,y in skeleton_points:
        if y < 0 or x < 0:
            continue
        xmin = max(round(x-l),0)
        xmin = int(xmin)
        xmax = min(xmin+4*2,xs)
        xmax = int(xmax)
        ymin = max(round(y-l),0)
        ymin = int(ymin)
        ymax = min(ymin+4*2,ys)
        ymax = int(ymax)
        #print(round(y+yh))

        
        #buggy numpy fix :D
        #print(ymin,ymax)
        print(xmin,xmax)
        
        tmp = image[ymin:ymax]
        tmp[:,xmin:xmax] = np.array([1,0,0])#np.array([99,99,99])
        
        #print(tmp)
        #image[ymin:ymax] = tmp
        #image[:,xmin:xmax] =1
        
    return image

img = draw_points(load_skeleton_data(index)[0],get_default(index))

plt.imshow(img)

# %%
max(2,3)
# %%
x = np.arange(9).reshape(3,3)
x[0:2,0:1]=1
# %%
x
# %%
def debug_show_image(skeleton_points,names, image):
    def plot_points(points,names):
        for i,(x,y) in enumerate(points):
            plt.plot(x,y, 'ro')
            plt.annotate(names[i][0], (x+2, y),backgroundcolor="w",size=3)
    plot_points(skeleton_points,names)
    plt.imshow(image)
    plt.savefig('filename.png', dpi=300)
    #plt.show(dpi=300)
index = 104
debug_show_image(*load_skeleton_data(index),get_default(index))

# %%

img = get_default(index)
data = load_skeleton_data(index)[0]


x,y = data[2].astype(int)

img[:,x:x+5]= 1


plt.imshow(img)
#%%


ys,xs,_ = img.shape

# %%

data[:,0] = - data[:,0] + xh
data[:,1] = - data[:,1] + yh

# %%
data
# %%
