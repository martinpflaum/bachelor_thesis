# About Licence:
Hello we used some code from https://github.com/NVlabs/stylegan2-ada-pytorch. I am not related to nvdia in any way. I wrote in every file that is my code under which licence it is. This project as a whole is licenced under the licence found in nvlicence.txt. In other words you can use my code under the MIT licence but not code from nvdia.

I changed some stuff in the nvdia code since i had problems with the jit compiler of the cuda code.

# About Code:
With '''
python script.py 
'''
the main program will be runned with different settings sequentially. The results will be automatically put in the results folder. In core.py is the main functionality. If you are looking how segmentation and normals should be preprocessed, look at data_utils/*. 

