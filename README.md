# About Licence:
Hello we used some code from https://github.com/NVlabs/stylegan2-ada-pytorch. I am not related to nvdia in any way. I wrote in every file that is my code under which licence it is. This project as a whole is licenced under the licence found in nvlicence.txt. In other words you can use my code under the MIT licence but not code from nvdia.

I changed some stuff in the nvdia code since i had problems with the jit compiler for the cuda code.

# About Code:
With '''
python script.py 
'''
the main program will be runned with different settings sequentially. You will need 2 pretrained GANs, that you obtained by running the code in https://github.com/NVlabs/stylegan2-ada-pytorch. They need to be called 1miocubes.pkl and scene_depth_gan_2000.pkl. With 1miocubes.pkl trained on https://www.kaggle.com/martinpflaum/one-million-cubes-depth and our combined image dataset. And scene_depth_gan_2000.pkl only trained on our combined image dataset.The results will be automatically put in the results folder. In core.py is the main functionality. 

In the experimental_and_data_vis folder is code that is not used by the main program. There i put some ideas that didn't make it into the final version and code for creating images (the creating images code only works if it is put the main folder). If you are looking how segmentation and normals should be preprocessed, look at data_utils/*. Some augmentations for our dataset can be found here https://github.com/martinpflaum/image-augmentation-with-point-clouds (not related to the project).
