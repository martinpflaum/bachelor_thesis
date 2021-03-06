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
import torch.nn as nn
import torch
import copy
from matplotlib import pyplot as plt
from data.data_utils import post_load_reverse_depth
import numpy as np
import torch
import torch.nn as nn
def noise_adjustment_local(x_org,model,var=0.3,num_iters=10,bs=10,l_func=nn.MSELoss()):
    model = model.eval()
    out = torch.zeros_like(x_org)
    x_org = x_org[None].repeat(bs,*[1 for _ in x_org.shape]) 
    #performance improvements by passing multiple batches at once
    y_target = model(x_org).detach()
    for k in range(num_iters):
        noise = (var**0.5) * torch.randn_like(x_org) # $\gamma  \sim \mathcal{N}(0,\,\sigma^{2})$
        x = x_org + noise # $x'=x+\gamma$  
        x = nn.Parameter(x)
        loss = l_func(model(x), y_target) # $loss = |model(x')-model(x)|_2^2 $
        loss.backward() #calculate the gradient
        g = x.grad.detach() 
        score = torch.abs(g) # $score = \left| \frac{d}{dx'}loss\right|$
        score = torch.mean(score,dim=0) #mean over batch dimension
        out += score*(1/num_iters) #mean over iterations
    out = out.cpu()
    n_dim = torch.prod(torch.tensor(out.size())) #number of dimensions
    norm = (n_dim**0.5)/torch.norm(out) #scaled normalization
    out = out*norm
    return out

def noise_adj_global(dset,model,var=0.3,num_iters=10,bs=10,l_func=nn.MSELoss(),device="cuda:0"):
    model = model.eval()
    model = model.to(device)
    func_args = (model,var,num_iters,bs,l_func) #for smaller line length

    xt,_ = dset[0]
    out = torch.zeros_like(xt).cpu().double()

    for i,(x_org,_) in enumerate(dset):
        x_org = x_org.to(device) #push the data to the correct device
        local_score = noise_adjustment_local(x_org,*func_args)
        #the *-operator will unpack the tuple and call the function with the arguments.
        out += local_score*(1/len(dset)) #mean over dset length
    n_dim = torch.prod(torch.tensor(out.size())) #number of dimensions
    norm = (n_dim**0.5)/torch.norm(out) #scaled normalization
    out = out*norm
    return out


def eval_importance(model,val_ds,importance,n_keep,loss_func=nn.MSELoss()):
    model = model.to("cuda:0")
    model = model.eval()
    input_dim = importance.shape[0]
    n_throw = input_dim - n_keep
    args = torch.argsort(importance.cpu()).numpy()
    args = args[::-1].copy()
    keep_args = args[:n_keep]
    throw_args = args[n_keep:]
    
    loss_out = 0

    for x_org,target in val_ds:
        x = copy.deepcopy(x_org)
        x_org = x_org.to("cuda:0")
        x[throw_args] = 0
        x = x
        x = x.to("cuda:0")
        #target = model(x_org[None]).detach()
        out = model(x[None]).detach()
        
        loss = loss_func(out, target.to("cuda:0"))
        loss = loss/len(val_ds)
        loss_out += loss

    loss_out = loss_out.detach().cpu().numpy()
    print("n_keep: ",n_keep," loss: ",loss_out)
    return f"n_keep: {n_keep} loss: {loss_out}\n"

def im_save_single(importance,n_keep,save_folder,learn,valid_ds,test_name,k,save_k):
    learn.model = learn.model.to("cuda:0")
    learn.model.eval()
    input_dim = importance.shape[0]
    n_throw = input_dim - n_keep
    args = torch.argsort(importance.cpu()).numpy()
    
    args = args[::-1].copy()
    args = torch.tensor(args)

    keep_args = args[:n_keep]
    

    throw_args = args[n_keep:]
    with torch.no_grad():
        input = valid_ds[k][0]
        input[throw_args] = 0#torch.randn_like(input[throw_args])
        pred_scene_depth = learn.model(input[None].to("cuda:0"))[0]
    img_vis = post_load_reverse_depth(pred_scene_depth)
    img_vis = np.clip(img_vis,0,1)
    if img_vis.shape[2] == 1:
        img_vis = img_vis[:,:,0]
    img = valid_ds[k][1]
    scene_depth = img[None]
    img_vis_target = post_load_reverse_depth(scene_depth[0])
    img_vis_target = np.clip(img_vis_target,0,1)
    if img_vis_target.shape[2] == 1:
        img_vis_target = img_vis_target[:,:,0]
    plt.imsave(f"{save_folder}/{test_name}_{save_k}.png",img_vis)
    plt.imsave(f"{save_folder}/target_{save_k}.png",img_vis_target)


def im_save_images(importance,n_keep,save_folder,learn,valid_ds,test_name,idx = [124,168,3,4]):
    for save_k,k in enumerate(idx): 
        im_save_single(importance,n_keep,save_folder,learn,valid_ds,test_name,k,save_k)