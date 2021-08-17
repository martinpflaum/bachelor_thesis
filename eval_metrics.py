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
        #print(f"iteration {z} of {num_iter}")
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

    rand_loss = rand_loss.detach().cpu().numpy()
    print("rand_loss",rand_loss)
    return f"rand_loss {rand_loss}\n"


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
        target = target.to("cpu")[None]#.reshape(1,1,64,64)
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

    top1 = top1.detach().cpu().numpy()
    top5 = top5.detach().cpu().numpy()
    top10 = top10.detach().cpu().numpy()
    print("top1",top1,"top5",top5,"top10",top10)
    return f"top1 {top1} top5 {top5} top10 {top10}\n"


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
    loss_out = loss_out.detach().cpu().numpy()
    print("constant loss of dset: ",loss_out)
    return f"constant loss of dset: {loss_out}\n"

def save_single(save_folder,learn,valid_ds,test_name,k,save_k,reverse_func):
    learn.model = learn.model.to("cuda:0")
    learn.model.eval()

    with torch.no_grad():
        input = valid_ds[k][0]
        pred_scene_depth = learn.model(input[None].to("cuda:0"))[0]
    img_vis = reverse_func(pred_scene_depth)
    if img_vis.shape[2] == 1:
        img_vis = img_vis[:,:,0]

    img = valid_ds[k][1]
    scene_depth = img[None]
    img_vis_target = reverse_func(scene_depth[0])
    if img_vis_target.shape[2] == 1:
        img_vis_target = img_vis_target[:,:,0]

    plt.imsave(f"{save_folder}/{test_name}_{save_k}.png",img_vis)
    plt.imsave(f"{save_folder}/target_{save_k}.png",img_vis_target)

def post_load_reverse_identity(img):
    img = ((img).cpu().permute(1,2,0))
    return np.array(img)

def save_images(save_folder,learn,valid_ds,test_name,idx = [124,168,3,4],reverse_func=post_load_reverse_identity):
    for save_k,k in enumerate(idx): 
        save_single(save_folder,learn,valid_ds,test_name,k,save_k,reverse_func)


def eval_model(val_ds,learn,loss_func=nn.MSELoss()):
    model = learn.model
    model = model.to("cuda:0")
    model = model.eval()
    
    loss_out = 0

    for x_org,target in val_ds:
        x_org = x_org.to("cuda:0")
        out = model(x_org[None]).detach()
        
        loss = loss_func(out, target.to("cuda:0"))
        loss = loss/len(val_ds)
        loss_out += loss

    loss_out = loss_out.detach().cpu().numpy()
    print("model val loss: ",loss_out)
    return f"model val loss: {loss_out}\n"

def run_all_metrics(save_folder,learn,valid_ds,file_name,reverse_func=post_load_reverse_identity):    
    save_images(save_folder,learn,valid_ds,file_name,reverse_func=reverse_func)
    out = ""
    out += get_mean_loss(valid_ds)
    out += calc_perc_metric(valid_ds,learn)
    out += calc_random_metric(valid_ds,learn)
    out += eval_model(valid_ds,learn)
    return out
