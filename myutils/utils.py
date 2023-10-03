import os
import torch
import pandas as pd
import numpy as np
import yaml
from collections import defaultdict


def normalize_tensor(x):
    """
    Compute mean and stddev of the **nonzero** elements of the event tensor
    we do not use PyTorch's default mean() and std() functions since it's faster
    to compute it by hand than applying those funcs to a masked array
    """
    # if (x != 0).sum() != 0:
    #     mean, stddev = x[x != 0].mean(), x[x != 0].std()
    #     x[x != 0] = (x[x != 0] - mean) / stddev
    nonzero = (x != 0)
    num_nonzeros = nonzero.sum()

    if num_nonzeros > 0:
        mean = x.sum() / num_nonzeros
        stddev = torch.sqrt((x ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero.float()
        x = mask * (x - mean) / stddev

    return x


def torch2cv2(image):
    """convert torch tensor to format compatible with cv2.imwrite"""
    image = torch.squeeze(image)  # H x W
    image = image.cpu().numpy() 
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)


def torch2frame(img):
    """
    img: torch.tensor, HxW or 1xHxW or CxHxW
    """
    img = img.squeeze()
    img = img.cpu().numpy().clip(0, 1)
    if len(img.shape) == 2:
        out_img = (img*255).astype(np.uint8)
    elif len(img.shape) == 3:
        out_img = (img.transpose(1, 2, 0)*255).astype(np.uint8)
    return out_img


class MetricTracker:
    def __init__(self, keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self._all_data = {}
        for key in keys:
            self._all_data[key] = []
        self.reset()
        
    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0
        for key in self._all_data.keys():
            self._all_data[key] = []

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]
        self._all_data[key].append(value)

    def avg(self, key):
        return self._data.average[key]
    
    def result(self):
        return dict(self._data.average)

    def all_data(self):
        return self._all_data


class Logger_yaml():
    def __init__(self, path):
        self.log_file = open(path, 'w')
        self.info_dict = defaultdict(list)

    def log_info(self, info: str):
        self.info_dict['info'].append(info)

    def log_dict(self, dict: dict, name: str):
        self.info_dict[name] = dict

    def __del__(self):
        yaml.dump(dict(self.info_dict), self.log_file)

