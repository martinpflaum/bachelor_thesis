#%%
import numpy as np
import os
import pickle
import nilearn
import nibabel as nib
from nilearn import plotting
import torch.nn as nn
import torch
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)
#BRAIN_DATA_ROOT = "D:/Datasets/BrainData"


BRAIN_FILE_NAME_TRAIN = "fir_sorted_v072_mean_tr1-2_z_1_train_WB.npy"
BRAIN_FILE_NAME_VAL = "fir_sorted_v072_mean_tr1-2_z_1_val_WB.npy"
BRAIN_FILE_NAME_TEST = "fir_sorted_v072_mean_tr1-2_z_1_test_WB.npy"

import numpy as np
import pickle
def save_pickle(input,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(input, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def load_reliability(brain_data_root,subject = 3):
    tmp = f"{brain_data_root}/fMRI_data/sub{subject:02d}/fir_sorted_v072_reliability_tr1-2_z_1_WB.pkl"
    return load_pickle(tmp)["test_reliability"]

def load_reliability_mask(brain_data_root,n = 8192,subject = 3):
    #2048
    #4096
    rel = load_reliability(brain_data_root,subject = subject)
    arg = np.argsort(rel)[::-1]
    return arg[:n]

def map2vec(data,reliability_mask):
    return data[reliability_mask]

def map2cube(data,mask_path,nan_indices_path):
    # initializing an array with dimensions of brain volume
    visual_mask_3D =  np.zeros((63, 76, 63))
    WB_voxel_mask = np.zeros(visual_mask_3D.shape)
    WB_voxel_mask_1D = np.zeros(visual_mask_3D.ravel().shape)
    
    # Loading data in non nan indices 
    nan_indices = np.load(nan_indices_path)
    mask = np.ones(WB_voxel_mask_1D.shape[0], dtype=bool)
    mask[nan_indices.astype(np.int)] = 0
    WB_voxel_mask_1D[mask] = data
    WB_nc = np.reshape(WB_voxel_mask_1D,WB_voxel_mask.shape)
    
    # applying brain mask and setting values outside brain mask = 0
    WB_mask = nib.load(mask_path)
    WB_mask = np.array(WB_mask.dataobj)  
    WB_nc[WB_mask==0]=0
    return WB_nc

class BrainLoader():
    def __init__(self,brain_data_root,file_name,mapToCupe = True,subject = 3,rel_n = 8192):
        self.brain_data_root = brain_data_root
        self.BRAIN_DATA_ARRAY = np.load(f"{brain_data_root}/fMRI_data/sub{subject:02d}/" + file_name)
        self.mapToCupe = mapToCupe
        if not self.mapToCupe:
            self.rel_mask = load_reliability_mask(brain_data_root,n = rel_n,subject=subject)
        


    def __call__(self,index,n=0,subject = 3):
        cube_data = None
        if self.mapToCupe:
            nan_indices_path = os.path.join(self.brain_data_root + '/fMRI_data',f"sub{subject:02d}",'z_1_nan_indicesWB.npy')
            mask_path = os.path.join(self.brain_data_root + '/fMRI_data',f"sub{subject:02d}",'mask.nii')
            cube_data = map2cube(self.BRAIN_DATA_ARRAY[index,n,:],mask_path, nan_indices_path)
            cube_data = torch.tensor(cube_data).float()
            cube_data = cube_data.reshape(1,63,76,63)
        else:
            cube_data = map2vec(self.BRAIN_DATA_ARRAY[index,n,:],self.rel_mask)
            cube_data = torch.tensor(cube_data).float()
        return cube_data


def get_mask_path(brain_data_root,subject = 3):
    mask_path = os.path.join(brain_data_root + '/fMRI_data',f"sub{subject:02d}",'mask.nii')
    return mask_path

def saveasnii(brain_mask,nii_save_path,nii_data):
    img = nib.load(brain_mask)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)




#saveasnii(mask_path,"./example.nii",cube_data)
#plotting.plot_glass_brain("./example.nii",colorbar=True)



#%%
#from brainmixer import PatchEmbed3d
"""bloader = BrainLoader(BRAIN_FILE_NAME_VAL,mapToCupe=False)
cube_data = bloader(0)
cube_data.shape

"""
# %%
#subject = 3
#np.load(f"D:/Datasets/BrainData/fMRI_data/sub03/fir_sorted_v072_mean_tr1-2_z_1_val_WB.npy")
# %%


# %%
# %%