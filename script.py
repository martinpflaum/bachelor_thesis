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
from core import run_all

#%%

def run_x5all(task,lr=None):
    #fastai default is 0.001
    args = {'lr':lr,'skip_training': False, 'train_gan': True, 'sub_folder' : None,'stable_loss': True, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)
    
    args = {'lr':lr,'skip_training': False, 'train_gan': False,'sub_folder' : None, 'stable_loss': False, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)


    args = {'lr':lr,'skip_training': False, 'train_gan': True, 'sub_folder' : None, 'stable_loss': False, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)


    args = {'lr':lr,'skip_training': False, 'train_gan': False, 'sub_folder' : None, 'stable_loss': True, 'calc_importance': True, 'input_dim': 4096, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)


    args = {'lr':lr,'skip_training': False, 'train_gan': False, 'sub_folder' : None, 'stable_loss': True, 'calc_importance': True, 'input_dim': 32768, 'task': task, 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
    run_all(args)



"""
args = {'skip_training': False, 'train_gan': False, 'sub_folder' : "1miocubes", 'stable_loss': True, 'calc_importance': True, 'input_dim': 4096, 'task': 'scene_depth', 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
args["pretrained_gen"] = "1miocubes.pkl"
run_all(args)

run_x5all(task = 'edges',lr=0.0002)

run_x5all(task = 'world_normal')

run_x5all(task = 'scene_depth')
"""
args = {'skip_training': False, 'train_gan': True, 'sub_folder' : "input_is_output_trgan(True)", 'stable_loss': False, 'calc_importance': False, 'input_dim': 4096, 'task': 'scene_depth', 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
args["make_input_depth"] = True
run_all(args)

args = {'skip_training': False, 'train_gan': False, 'sub_folder' : "input_is_output_trgan(False)", 'stable_loss': False, 'calc_importance': False, 'input_dim': 4096, 'task': 'scene_depth', 'pretrained_gen': None, 'learn_save_file': None, 'file_name': None, 'results_folder': 'results'}
args["make_input_depth"] = True
run_all(args)

