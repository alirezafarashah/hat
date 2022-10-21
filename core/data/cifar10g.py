import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np


class GENERATIVECIFAR10(torch.utils.data.Dataset):

    def __init__(self, each_class=5000):
        self.data = torch.Tensor(np.load("/kaggle/working/cifar10_training_gen_data.npy"))
        self.data = self.data / 255
        self.target = torch.Tensor([i // each_class for i in range(len(self.data))], dtype=torch.long)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return self.data[index], self.target[index]


# DATA_DESC = {
#     'data': 'cifar10',
#     'classes': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
#     'num_classes': 10,
#     'mean': [0.4914, 0.4822, 0.4465],
#     'std': [0.2023, 0.1994, 0.2010],
# }


def load_cifar10_generative(data_dir=None, use_augmentation=False, each_class=5000):
    """
    Returns CIFAR10 train, test datasets and dataloaders.
    Arguments:
        data_dir (str): path to data directory.
        use_augmentation (bool): whether to use augmentations for training set.
    Returns:
        train dataset, test dataset.
    """

    return GENERATIVECIFAR10()
