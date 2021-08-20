#%%
from core import run_all

#%%

def run_x5all(task):
    args = {'skip_training': False, 'train_gan': True, 'sub_folder' : None,'stable_loss': True, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)
    
    args = {'skip_training': False, 'train_gan': False,'sub_folder' : None, 'stable_loss': False, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)


    args = {'skip_training': False, 'train_gan': True, 'sub_folder' : None, 'stable_loss': False, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)


    args = {'skip_training': False, 'train_gan': False, 'sub_folder' : None, 'stable_loss': True, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)


    args = {'skip_training': False, 'train_gan': False, 'sub_folder' : None, 'stable_loss': True, 'calc_importance': True, 'input_dim': 32768, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)

args = {'skip_training': False, 'train_gan': False, 'sub_folder' : "1miocubes", 'stable_loss': True, 'calc_importance': True, 'input_dim': 32768, 'task': 'scene_depth', 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
args["pretrained_gen"] = "1miocubes.pkl"
run_all(args)

#run_x5all(task = 'world_normal')


run_x5all(task = 'scene_depth')
run_x5all(task = 'edges')


# %%

