import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.datasets import MNIST


def dict2namespace(config_dict):
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def split_and_save_mnist(root='../../data/mnist'): 
    try:
        os.makedirs(os.path.join(root, 'splitted'), exist_ok=False)
    except:
        return

    mnist = MNIST(
        root=root,
        train=True,
        download=True,
        transform=T.Compose([
            T.ToTensor(),
            T.Normalize((0.5), (0.5)),
        ])
    )
    loader = torch.utils.data.DataLoader(
        mnist, batch_size=1000, shuffle=False,
        num_workers=0, drop_last=False,
    )

    small_sets = {label: [] for label in range(10)}
    for x, y in tqdm(loader):
        for i, label in enumerate(y):
            small_sets[label.item()].append(x[i].unsqueeze(0))
    
    for label in small_sets.keys():
        small_sets[label] = torch.cat(small_sets[label], dim=0).numpy()
        np.savez(
            os.path.join(root, 'splitted', f'{label}.npz'),
            data=small_sets[label]
        )

            


