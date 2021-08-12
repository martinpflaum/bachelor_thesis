#%%
import torch
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
    
def calc_random_metric(valid_ds,learn,num_iter=10,loss_func = nn.MSELoss()):
    learn.model = learn.model.to("cuda:0")
    learn.model.eval()

    rand_loss = 0
    num_iter = 10
    idx = np.arange(len(valid_ds))
    for z in range(num_iter):
        print(f"iteration {z} of {num_iter}")
        loss_one_iter = 0
        np.random.shuffle(idx)
        for k in range(len(valid_ds)): 
            with torch.no_grad():
                input =valid_ds[k][0]
                pred_scene_depth = learn.model(input[None].to("cuda:0"))[0]
                target = valid_ds[idx[k]][1]
                target = target.to("cuda:0")
                loss = loss_func(pred_scene_depth,target)
                loss = loss/len(valid_ds)
                loss_one_iter += loss
        loss_one_iter = loss_one_iter / num_iter
        rand_loss += loss_one_iter
    print("rand_loss",rand_loss)


def calc_perc_metric(valid_ds,learn):
    learn.model = learn.model.to("cuda:0")
    learn.model.eval()
    preds = []
    for k in range(len(valid_ds)): 
        with torch.no_grad():
            input = valid_ds[k][0]
            pred_scene_depth = learn.model(input[None].to("cuda:0"))[0].to("cpu")
            preds += [pred_scene_depth[None]]


    preds = torch.cat(preds,dim = 0)
    #print(preds.shape)
    N = preds.shape[0]
    top1 = 0
    top5 = 0
    top10 = 0
    for k in range(len(valid_ds)): 
        target = valid_ds[k][1]
        target = target.to("cpu").reshape(1,1,64,64)
        #print(target.shape)
        diff = torch.mean(((preds-target)**2).reshape(N,-1),dim = -1)
        #print(diff.shape)
        argmins = torch.argsort(diff)
        argmins = list(argmins)
        percentil = argmins.index(k)
        
        
        if percentil < 10:
            top10 += 1/len(valid_ds)
        if percentil < 5:
            top5 += 1/len(valid_ds)
        if percentil < 1:
            top1 += 1/len(valid_ds)

    print("top1",top1,"top5",top5,"top10",top10)


from data.data_utils import post_load_reverse_depth

def get_mean(valid_ds):
    out = torch.zeros_like(valid_ds[0][1])
    for k in range(len(valid_ds)): 
        with torch.no_grad():
            target = valid_ds[k][1]
            target = target / len(valid_ds)
            out += target
    return out
def get_mean_loss(valid_ds,loss_func = nn.MSELoss()):
    mean = get_mean(valid_ds)
    loss_out = 0
    for k in range(len(valid_ds)): 
        with torch.no_grad():
            target = valid_ds[k][1]
            loss = loss_func(target,mean)
            loss = loss / len(valid_ds)
            loss_out += loss
    print("constant loss of dset: ",loss_out)

def save_single(save_folder,learn,valid_ds,test_name,k,save_k):
    learn.model = learn.model.to("cuda:0")
    learn.model.eval()

    with torch.no_grad():
        input = valid_ds[k][0]
        pred_scene_depth = learn.model(input[None].to("cuda:0"))[0]
    img_vis = post_load_reverse_depth(pred_scene_depth)[:,:,0]

    img = valid_ds[k][1]
    scene_depth = img[None]
    img_vis_target = post_load_reverse_depth(scene_depth[0])[:,:,0]

    plt.imsave(f"{save_folder}/{test_name}_{save_k}.png",img_vis)
    plt.imsave(f"{save_folder}/target_{save_k}.png",img_vis_target)


def save_images(save_folder,learn,valid_ds,test_name,idx = [124,168,3,4]):
    for save_k,k in enumerate(idx): 
        save_single(save_folder,learn,valid_ds,test_name,k,save_k)


def run_all_metrics(save_folder,learn,valid_ds,file_name):    
    save_images(save_folder,learn,valid_ds,file_name)
    get_mean_loss(valid_ds)
    calc_perc_metric(valid_ds,learn)
    calc_random_metric(valid_ds,learn)

