import torch
import warnings
warnings.simplefilter("ignore", UserWarning)
#from perceiver_pytorch import Perceiver
import torchvision
import numpy as np
import PIL
import PIL.Image
import pandas as pd
import scipy
import pickle

import scipy
import scipy.spatial

def save_pickle(input,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(input, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)


def open_image(name,view_mode,data_set_root,frame = 0):#D:/Datasets/ImageDataset
    """
    {"world_normal","scene_depth","lightning","reflectance","default"}
    """
    #index +1
    return (PIL.Image.open( data_set_root+"/"+f"{name}/{view_mode}_f_{frame}.png")).convert('RGB')

def post_load_default(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255
    img = torchvision.transforms.functional.to_tensor(img)
    img = torchvision.transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],False)
    return img

def post_load_scene_depth(img):
    img = np.array(img)
    img = img[:,:,0] + img[:,:,1] * 256
    img = img.astype(np.float32)
    img = img.reshape([img.shape[0],img.shape[1],1])
    maximum_dist = 256 * 4#256*3#

    #normalize

    #img = torchvision.transforms.functional.to_tensor((img / (maximum_dist / 2) )  - 1)
    img = torchvision.transforms.functional.to_tensor((img / (maximum_dist) ))
    
    
    return img


def get_rotation_matrix(folder_name,dataset_root):
    df = pd.read_csv(f"{dataset_root}/{folder_name}/info.csv")[["camera_rotation_pitch","camera_rotation_yaw"]]
    df = np.array(df)[0]
    #pitch y 
    #yaw z
    #roll x 
    pitch = df[0]
    yaw = df[1]
    r = scipy.spatial.transform.Rotation.from_euler('zyx', [yaw, 0, 0], degrees=True)
    r = r.as_matrix()
    return r


def post_load_normal(img,folder_name,dataset_root):
    img = np.array(img)
    img = (img.astype(np.float32) / 255) * 2 - 1 #normalize
    
    rot_matrix = get_rotation_matrix(folder_name,dataset_root)
    img = (torch.tensor(img) @ torch.tensor(rot_matrix).float()).numpy()
    img = (img + 1)/2
    img = torchvision.transforms.functional.to_tensor(img)
    return img


def post_load_reverse_normal(input):
    input = input[None]
    qn = torch.norm(input, p=2, dim=1).unsqueeze(dim=1) + 1e-12
    input = input.div(qn)[0]
    img= ((input+1).cpu().permute(1,2,0))/ (2)
    return np.array(img)

def post_load_reverse_lightning(input):
    img= ((input+1).cpu().permute(1,2,0))/ (2)
    return np.array(img)


def post_load_lightning(img):
    img = np.array(img)
    #img = (img[:,:,0] + img[:,:,1]+ img[:,:,2])/3
    img = (img.astype(np.float32) / 255) * 2 - 1 #normalize
    img = torchvision.transforms.functional.to_tensor(img)
    return img

def post_load_reverse_depth(img):
    #img= ((img+1).cpu().permute(1,2,0))/ (2/256 * 3)
    img= ((img).cpu().permute(1,2,0))#/ (1/256 * 3)
    
    return np.array(img)

    #((y_pred[0]+1).cpu().permute(1,2,0))/ (2/256 * 3) )
from scipy.ndimage.filters import convolve, gaussian_filter


def canny_edge(im):
    im = np.array(im, dtype=float) #Convert to float to prevent clipping values

    #Gaussian blur to reduce noise
    im2 = gaussian_filter(im, 3)
    #Use sobel filters to get horizontal and vertical gradients
    im3h = convolve(im2,[[-1,0,1],[-2,0,2],[-1,0,1]])
    im3v = convolve(im2,[[1,2,1],[0,0,0],[-1,-2,-1]])

    #Get gradient and direction
    grad = np.power(np.power(im3h, 2.0) + np.power(im3v, 2.0), 0.5)
    theta = np.arctan2(im3v, im3h)
    #thetaQ = (np.round(theta * (5.0 / np.pi)) + 5) % 5 #Quantize direction
    return grad

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def load_edge_tensor(name,data_set_root,frame = 0):
    img = np.array(open_image(name,"default",data_set_root,frame = frame)).astype(np.float32)
    img = rgb2gray(img)
    img = canny_edge(img)
    img = img.reshape(img.shape[0],img.shape[1],1)
    img = (img.astype(np.float32) / 255) #* 2 - 1 #normalize
    img = torchvision.transforms.functional.to_tensor(img)
    return img


def get_train_val_test_split(file_name):#"train_test_split.csv"
    df = pd.read_csv(file_name)
    df = df[["name","train_val_test"]]
    split = np.array(df)

    train = split[:,1] == 0
    val = split[:,1] == 1
    test = split[:,1] == 2

    train = split[train][:,0]
    val = split[val][:,0]
    test = split[test][:,0]

    return train,val,test


"""
from os import listdir
from os.path import isfile, join

org = "D:/Datasets/BrainData/rendered_animations_final/"
def check_index(index):
    mypath =org+ f"Anim_{(index+1):05d}"

    #f"Anim_{(index):05d}/{view_mode}_f_{frame}.png"
    view_modes = ['scene_depth','world_normal', 'lightning','objects',"reflectance","default"]#"edges":1.5 "edges" : nn.L1Loss() ,"reflectance":nn.L1Loss()
    frames = range(40) 

    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for viewmode in view_modes:
        for frame in frames:
            if not f"{viewmode}_f_{frame}.png" in onlyfiles:
                print(f"{viewmode}_f_{frame}.png is not in ANIM "+str(index+1))
    

    for frame in frames:
        if not f"skeleton_positions_screen_f_{frame}.csv" in onlyfiles:
            print(f"skeleton_positions_screen_f_{frame}.csv is not in ANIM"+str(index+1))
    
        if not f"skeleton_positions_world_f_{frame}.csv" in onlyfiles:
            print(f"skeleton_positions_world_f_{frame}.csv is not in ANIM"+str(index+1))
    
    if not f"objects_mapper.csv" in onlyfiles:
        print(f"objects_mapper.csv is not in ANIM"+index+1)
    
    if not f"info.csv" in onlyfiles:
        print(f"info.csv is not in ANIM"+index+1)
    
        #skeleton_positions_screen_f_1
        #skeleton_positions_world_f_36

for k in range(725,1440):
    check_index(k)
"""