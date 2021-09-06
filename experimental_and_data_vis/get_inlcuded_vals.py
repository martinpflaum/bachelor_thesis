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
