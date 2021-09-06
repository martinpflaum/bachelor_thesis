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


def get_flow(index):
    folder_name = f"Anim_{index+1:05}"
    img = open_image(folder_name,"flow",dataset_root)
    img = np.array(img)/255
    img = np.clip(img,0,1)
    return img


def get_depth(index):
    folder_name = f"Anim_{index+1:05}"
    img = open_image(folder_name,"scene_depth",dataset_root)
    img = np.array(img)
    img = img[:,:,0] + img[:,:,1] * 256
    img = img.astype(np.float32)
    img = img.reshape([img.shape[0],img.shape[1],1])
    maximum_dist = 256 * 4#256*3#

    #normalize

    #img = torchvision.transforms.functional.to_tensor((img / (maximum_dist / 2) )  - 1)
    img = img / (maximum_dist) 
    return img[:,:,0]


# %%
def vert_grid(func,idx):
    out = []
    for k in idx:
        out += [func(k)]
    return np.concatenate(out,axis=1)


def save_images(func):
    img0 = vert_grid(func,[104,1004,509])
    img1 = vert_grid(func,[304,1406,3510])

    out = [img0,img1]
    img = np.concatenate(out,axis=0)

    plt.imsave(func.__name__[4:]+".png",img.astype(float))

funcs = [get_default,get_edges,get_flow,get_lightning,get_normal,get_reflectance,get_seg_img,get_depth]
for func in funcs:
    save_images(func)

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
import PIL
import PIL.Image as Image
dset_root = "D:/one_million_cubes"
k = 4
def get_million(k):
    file_name = f"scene_depth_rand_{k:07}.png"
    return np.array(Image.open(f"{dset_root}/{file_name}"))




def save_images(func):
    out = []
    out += [vert_grid(func,[27,8048,9004,12509])]
    out += [vert_grid(func,[9274,268,1206,13510])]
    out += [vert_grid(func,[499,7024,1496,11510])]
    out += [vert_grid(func,[9876,30224,1003,32510])]
    
    
    img = np.concatenate(out,axis=0)

    plt.imsave(func.__name__[4:]+".png",img.astype(float))

save_images(get_million)


# %%
