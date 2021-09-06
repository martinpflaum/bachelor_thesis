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
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
import random
from torchvision.transforms import functional as F

class LRandomResizedCrop(transforms.RandomResizedCrop):
    """
    this code is originaly from https://github.com/kenshohara/3D-ResNets-PyTorch
    (with some changes) it was published under MIT License
    """
    def __init__(self,
                 size,
                 scale=(0.5, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
       
    def __call__(self, input):
        self.random_crop = self.get_params(input[0], self.scale, self.ratio)
        top, left, height, width = self.random_crop
        out = []
        for img in input:
            out += [F.resized_crop(img,top, left, height, width, self.size, self.interpolation)]
        return out

class LRandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        """
        this code is originaly from https://github.com/kenshohara/3D-ResNets-PyTorch
        (with some changes) it was published under MIT License
        """
        super().__init__(p)
            
    def __call__(self, input):
        random_p = random.random()
        out = []
        if random_p < self.p:
            for img in input:
                out += [F.hflip(img)]
        else:
            out = input
        return out

class LResize():
    def __init__(self,size, interpolation=InterpolationMode.BILINEAR):
        self.func = transforms.Resize(size, interpolation)
    def __call__(self,input):
        out = []
        for img in input:
            out += [self.func(img)]
        return out


class LCenterCrop():
    def __init__(self,size):
        self.func = transforms.CenterCrop(size)
    def __call__(self,input):
        out = []
        for img in input:
            out += [self.func(img)]
        return out


class LCompose():
    def __init__(self,funcs):
        self.funcs = funcs
    def __call__(self,input):
        for func in self.funcs:
            input = func(input)
        return input
