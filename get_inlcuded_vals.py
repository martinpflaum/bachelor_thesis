#%%
import os
import numpy as np
from data.data_utils import save_pickle
from data.brainloading import load_reliability_mask,map2cube
import nibabel as nib
from nilearn import plotting

n_input = 32768
def saveasnii(brain_mask,nii_save_path,nii_data):
    img = nib.load(brain_mask)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)

brain_data_root = "D:/Datasets/BrainData"
subject = 3
rel_mask = load_reliability_mask(brain_data_root,n = n_input,subject = subject)
out = np.zeros((100045))
importance = np.ones((n_input))
out[rel_mask] = importance

nan_indices_path = os.path.join(brain_data_root + '/fMRI_data',f"sub{subject:02d}",'z_1_nan_indicesWB.npy')
mask_path = os.path.join(brain_data_root + '/fMRI_data',f"sub{subject:02d}",'mask.nii')
                    
cube_data = map2cube(out,mask_path, nan_indices_path)
saveasnii(mask_path,"./temp.nii",cube_data)
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
nilearn_cmaps.keys()
#%%
view = plotting.view_img_on_surf("./temp.nii",surf_mesh='fsaverage',cmap=nilearn_cmaps["bwr"]) 
view.save_as_html("./included_vals.html") 

# %%
plotting.plot_glass_brain("./temp.nii",colorbar=True,cmap=nilearn_cmaps["bwr"],output_file="./included_vals.png")
# %%
