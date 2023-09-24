import time
import os
import random
import numpy as np
import torch
import torch.utils.data

from utils import read_image

class CS4243dataset(torch.utils.data.Dataset):
    """
    Load WBC dataset
    """
    def __init__(self, path_to_folder, label_dict, is_train, transform):
        self.image_paths = []
        self.labels = []
        self.is_train = is_train
        self.transform = transform
        
        for label_name in label_dict.keys():
            if is_train:
                img_folder_path = os.path.join(path_to_folder, 'train', label_name)
            else:
                img_folder_path = os.path.join(path_to_folder, 'test', label_name)

            for f in os.listdir(img_folder_path):
                img_path = os.path.join(img_folder_path, f)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label_dict[label_name])
        
    def __getitem__(self, index):
        img = read_image(self.image_paths[index])
        label = self.labels[index]
        
        return self.transform(img), label
    
    def __len__(self):
        return len(self.labels)
