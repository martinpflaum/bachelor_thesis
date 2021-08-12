# THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USEd  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE 
# THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USEd  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE 
# THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USEd  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE 
# THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE d THIS IS NOT USEd  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE 
class BrainDataset(Dataset):
    def __init__(self,indicies,brain_data_file,data_set_root,is_train):
        #11520
        #2879
        self.data_set_root = data_set_root
        self.segMaskLoader = SegMaskLoader("version2.1.pickle",self.data_set_root)

        self.is_train = is_train

        self.size = len(indicies)
        if is_train:
            self.size = self.size * 3

        self.indicies = indicies

        img_size = 224
        self.y_transform = torchvision.transforms.Resize((img_size,img_size))
        self.segmentation_transform = torchvision.transforms.Resize((img_size,img_size),interpolation = InterpolationMode("nearest"))
        
        self.brain_loader = BrainLoader(brain_data_file)

    def __len__(self):
        return self.size
    def __getitem__(self, index):
        index = self.indicies[index]
        #task_num_output = {'scene_depth': 1, 'world_normal': 3, 'lightning': 3}
        x = None
        if self.is_train:

            #x = self.brain_loader(bindex,n)
            """alpha = nn.functional.softmax(torch.rand(3,1,1,1))
            x = []
            x += [self.brain_loader(bindex,0)]
            x += [self.brain_loader(bindex,1)]
            x += [self.brain_loader(bindex,2)]
            x = torch.sum(alpha*torch.cat(x,dim=0),dim=0)[None]"""

        else:
            x = self.brain_loader(index)
        scene_depth = self.y_transform(post_load_scene_depth(open_image(index,"scene_depth",self.data_set_root)))
        world_normal = self.y_transform(post_load_normal(open_image(index,"world_normal",self.data_set_root)))
        lightning = self.y_transform(post_load_lightning(open_image(index,"lightning",self.data_set_root)))
        segmentation = self.segmentation_transform(torchvision.transforms.functional.to_tensor(self.segMaskLoader(index)))
        edges = self.y_transform(load_edge_tensor(index,self.data_set_root))
        reflectance = self.y_transform(post_load_default(open_image(index,"reflectance",self.data_set_root)))
      
      
        y = torch.cat([scene_depth,world_normal,lightning,segmentation,edges,reflectance],dim = 0)

        return x,y

# THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE  THIS IS NOT USE 