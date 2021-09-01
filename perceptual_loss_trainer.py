#%%
import torchvision
import math
import torch
import torch.nn as nn
import numpy as np
import lightly
import pandas as pd
from data.data_utils import get_train_val_test_split,open_image,\
    post_load_default,post_load_scene_depth,post_load_normal,\
    post_load_lightning,load_edge_tensor,load_pickle,save_pickle
from perceptual_loss import PerceptualLoss
from torch.utils.data import Dataset
class GaussianNoise:
    """Applies random Gaussian noise to a tensor.

    The intensity of the noise is dependent on the mean of the pixel values.
    See https://arxiv.org/pdf/2101.04909.pdf for more information.

    """

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        mu = sample.mean()
        snr = np.random.randint(low=4, high=8)
        sigma = mu / snr
        noise = torch.normal(torch.zeros(sample.shape), sigma)
        return sample + noise

def depth_calc_std_mean(img_data_set_root):
    train,val,test = get_train_val_test_split("./data_splits/train_test_split.csv")
    out = []
    for name in train:
        scene_depth = post_load_scene_depth(open_image(name,"scene_depth",img_data_set_root)).reshape(-1)
        out += [scene_depth]
    out = torch.cat(out).reshape(-1)
    return torch.std_mean(out, unbiased=False)


def get_all(file_name):
    df = pd.read_csv(file_name)
    df = df[["train_val_test"]]
    split = np.array(df)
    return split
class BrainDatasetSceneDepth(Dataset):
    def __init__(self,img_data_set_root,indicies) :
        super().__init__()
        self.img_data_set_root = img_data_set_root
        self.indicies = indicies
        self.size = len(indicies)
    def __len__(self):
        #print("get_len")
        return self.size
    def __getitem__(self, index):
        name = self.indicies[index]
        return post_load_scene_depth(open_image(name,"scene_depth",self.img_data_set_root))
img_data_set_root="D:/ImageDatasetBig"
indicies = get_all("./data_splits/train_test_split.csv")
indicies
#%%
dset = BrainDatasetSceneDepth(img_data_set_root,indicies)

#%%
num_workers = 0
batch_size = 128
seed = 1
epochs = 50
input_size = 64

# dimension of the embeddings
num_ftrs = 512
# dimension of the output of the prediction and projection heads
out_dim = proj_hidden_dim = 512
# the prediction head uses a bottleneck architecture
pred_hidden_dim = 128
# use 2 layers in the projection head
num_mlp_layers = 2

mean,std = torch.tensor(0),torch.tensor(1)
mean,std = depth_calc_std_mean(img_data_set_root)

mean,std = mean.item(),std.item()
mean,std = (mean,mean,mean),(std,std,std)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.RandomResizedCrop(size=(64,64), scale=(0.2, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.GaussianBlur(21),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean,std),
    GaussianNoise(),
])

collate_fn = lightly.data.BaseCollateFunction(transform)

torch.manual_seed(0)
np.random.seed(0)

# set the path to the dataset
path_to_data = 'C:/Users/Martin/Downloads/test'
dataset_train_simsiam = lightly.data.LightlyDataset(
    input_dir=path_to_data
)
dataloader_train_simsiam = torch.utils.data.DataLoader(
    dataset_train_simsiam,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])

# create the SimSiam model using the backbone from above
model = lightly.models.SimSiam(
    backbone,
    num_ftrs=num_ftrs,
    proj_hidden_dim=pred_hidden_dim,
    pred_hidden_dim=pred_hidden_dim,
    out_dim=out_dim,
    num_mlp_layers=num_mlp_layers
)

# SimSiam uses a symmetric negative cosine similarity loss
criterion = lightly.loss.SymNegCosineSimilarityLoss()

# scale the learning rate
lr = 0.05 * batch_size / 256
# use SGD with momentum and weight decay
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=lr,
    momentum=0.9,
    weight_decay=5e-4
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

avg_loss = 0.
avg_output_std = 0.

for e in range(epochs):

    for (x0, x1), _, _ in dataloader_train_simsiam:

        # move images to the gpu
        x0 = x0.to(device)
        x1 = x1.to(device)

        # run the model on both transforms of the images
        # the output of the simsiam model is a y containing the predictions
        # and projections for each input x
        y0, y1 = model(x0, x1)

        # backpropagation
        loss = criterion(y0, y1)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output, _ = y0
        output = output.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()

        # use moving averages to track the loss and standard deviation
        w = 0.9
        avg_loss = w * avg_loss + (1 - w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

    # the level of collapse is large if the standard deviation of the l2
    # normalized output is much smaller than 1 / sqrt(dim)
    collapse_level = max(0., 1 - math.sqrt(out_dim) * avg_output_std)
    # print intermediate results
    print(f'[Epoch {e:3d}] '
        f'Loss = {avg_loss:.2f} | '
        f'Collapse Level: {collapse_level:.2f} / 1.00')




model = PerceptualLoss(model.backbone.cpu(),mean,std).cpu()
save_pickle(model,"perceptual_loss.pth")
# %%
