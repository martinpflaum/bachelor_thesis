#%%
import pandas as pd
import PIL
import numpy as np
import PIL.Image
import pickle


def save_pickle(input,filename):
    with open(filename, 'wb') as handle:
        pickle.dump(input, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

def html_to_rgb(input):
    #https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    input = input.lstrip('#')
    return tuple(int(input[i:i+2], 16) for i in (0, 2, 4))

def create_colors(input):
    out = []
    for elem in input:
        out.append(html_to_rgb(elem))
    return np.array(out)

#https://mokole.com/palette.html
distinct_colors = create_colors("""#4682b4
#000080
#d2691e
#32cd32
#8fbc8f
#8b008b
#808080
#228b22
#800000
#808000
#483d8b
#008b8b
#b03060
#ff4500
#ffa500
#ffff00
#00ff00
#9400d3
#dc143c
#00ffff
#0000ff
#adff2f
#ff00ff
#1e90ff
#fa8072
#add8e6
#ff1493
#7b68ee
#ee82ee
#98fb98
#ffe4b5
#ffb6c1""".split("\n"))

def get_color_to_instance_dict(folder_name,dataset_root):
    """
    Please ignore objectClass, its deprecated.
    """
    global_mapper = {255:"human",
                    250:"h_phone",
                    253:"h_laptop",
                    254:"h_bottle",
                    251:"h_cup",
                    249:"h_guitar",
                    0:"Level2Walls"}
    df = pd.read_csv(f"{dataset_root}/{folder_name}/objects_mapper.csv")[["objectInstanceID","displayNames"]]
    df["objectInstanceID"] = df["objectInstanceID"] % 1000 #this is historical.
    out = np.array(df)
    mapper = dict(out)
    mapper.update(global_mapper)
    return mapper

def open_image(folder_name,view_mode,dataset_root,frame = 0):
    return (PIL.Image.open( f"{dataset_root}/{folder_name}/{view_mode}_f_{frame}.png")).convert('RGB')

def get_instance_to_class_dict(class_to_instance):
    out = {}
    for elem in class_to_instance.keys():
        for xelem in class_to_instance[elem]:
            out[xelem] = elem 

    return out

def get_class_to_idx_dict(class_to_instance):
    out = {}
    for i,elem in enumerate(class_to_instance.keys()):
        out[elem] = i
    return out
#labelssmoothing with gaussian n
import torchvision
class SegMaskLoader():
    def __init__(self,class_to_instance_file,dataset_root):
        super().__init__()
        class_to_instance = load_pickle(class_to_instance_file)
        #"human","h_phone","h_laptop","h_bottle","h_cup","h_guitar"
        self.class_to_instance = class_to_instance
        self.class_to_idx = get_class_to_idx_dict(self.class_to_instance)
        self.num_classes = len(self.class_to_idx)
        self.instance_to_class = get_instance_to_class_dict(class_to_instance)
        
        self.dataset_root = dataset_root
    def __call__(self,folder_name):
        color_to_instance = get_color_to_instance_dict(folder_name,self.dataset_root)
        obj = np.array((open_image(folder_name,"objects",self.dataset_root)))[:,:,0]

        out = np.zeros((*obj.shape,self.num_classes))
        for color in np.unique(obj):
            if not color in list(color_to_instance.keys()):
                raise RuntimeError("Didn t found color:",color)
            idx = self.class_to_idx[self.instance_to_class[color_to_instance[color]]]
            out[obj == color,idx] = 1 
        
        return out

def seg_mask_to_color(input):
    return distinct_colors[np.argmax(input,axis = 2)]


def get_object_excluded(img,obj,idx):
    img = img.copy()
    shape = img.shape
    idx = obj.reshape(-1) != idx
    img = img.reshape(shape[0] * shape[1],3)
    img[idx,:] = 0
    return img.reshape(*shape)

"""
from seg_mask_loader import SegMaskLoader
from seg_mask_loader import seg_mask_to_color
import matplotlib.pyplot as plt
segmask_loader = SegMaskLoader("version1.pt")
out = segmask_loader(7)
plt.imshow(seg_mask_to_color(out))
"""
#ximg = get_object_excluded(img,obj,20)
# %%
