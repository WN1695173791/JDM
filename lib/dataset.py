import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST, CIFAR10


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
    loader = DataLoader(
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


def split_and_save_cifar10(root='../../data/cifar10'): 
    try:
        os.makedirs(os.path.join(root, 'train', 'splitted'), exist_ok=False)
        os.makedirs(os.path.join(root, 'test', 'splitted'), exist_ok=False)
    except:
        return

    for mode in ['train', 'test']:
        is_train = True if mode == 'train' else False
        cifar10 = CIFAR10(
            root=os.path.join(root, mode),
            train=is_train,
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        loader = DataLoader(
            cifar10, batch_size=1000, shuffle=False,
            num_workers=0, drop_last=False,
        )

        small_sets = {label: [] for label in range(10)}
        for x, y in tqdm(loader):
            for i, label in enumerate(y):
                small_sets[label.item()].append(x[i].unsqueeze(0))
        
        for label in small_sets.keys():
            small_sets[label] = torch.cat(small_sets[label], dim=0).numpy()
            np.savez(
                os.path.join(root, mode, 'splitted', f'{label}.npz'),
                data=small_sets[label]
            )


def inf_mnist_loop(
    num: int,
    splitted_root: str = '../../data/mnist/splitted',
):
    data = np.load(os.path.join(splitted_root, f'{num}.npz'))['data']
    while True:
        i = np.random.randint(data.shape[0])
        yield data[i]


def inf_cifar10_loop(
    num: int,
    splitted_root: str = '../../data/cifar10/train/splitted',
):
    data = np.load(os.path.join(splitted_root, f'{num}.npz'))['data']
    while True:
        i = np.random.randint(data.shape[0])
        yield data[i]


def int_to_mnist(labels, mnist_loopers):
    assert len(labels.shape) == 1 # shape [B,]
    result = []
    for label in labels:
        temp = torch.Tensor(
            next(mnist_loopers[label.item()]),
        ).unsqueeze(0)
        temp = F.pad(temp, (2, 2, 2, 2), 'constant')
        result.append(temp)
    return torch.cat(result, dim=0)


def int_to_cifar10(labels, cifar10_loopers):
    assert len(labels.shape) == 1 # shape [B,]
    result = []
    for label in labels:
        temp = torch.Tensor(
            next(cifar10_loopers[label.item()]),
        ).unsqueeze(0)
        result.append(temp)
    return torch.cat(result, dim=0)

            
def DataLooper(config, batch_size):
    if config.dataset.x_name.lower() == 'cifar10':
        # Load dataset and dataloader
        dataset = CIFAR10( root=os.path.join(config.dataset.x_root, 'train'),
            train=True,
            download=False,
            transform=T.Compose([
                T.RandomHorizontalFlip(0.5),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.train.n_workers,
            drop_last=True,
        )

        if config.dataset.y_name.lower() == 'onehot':
            assert 0==1

        elif config.dataset.y_name.lower() == 'mnist':
            split_and_save_mnist(config.dataset.y_root)
            mnist_loopers = []

            for i in range(10):
                looper = inf_mnist_loop(i)
                mnist_loopers.append(looper)

            while True:
                for x, y in iter(dataloader):
                    y = int_to_mnist(y, mnist_loopers)
                    yield x, y

        elif config.dataset.y_name.lower() == 'anti_cifar10':
            split_and_save_cifar10(config.dataset.y_root)
            cifar10_loopers = []

            for i in range(10):
                looper = inf_cifar10_loop(i)
                cifar10_loopers.append(looper)

            while True:
                labels = np.random.randint(0, 5, size=batch_size)
                labels = torch.tensor(labels, dtype=torch.long)
                anti_labels = 9 - labels
                x = int_to_cifar10(labels, cifar10_loopers)
                y = int_to_cifar10(anti_labels, cifar10_loopers)
                yield x, y
                
    elif config.dataset.x_name.lower() == 'mnist':

        if config.dataset.y_name.lower() == 'anti_mnist':
            split_and_save_mnist(config.dataset.y_root)
            mnist_loopers = []

            for i in range(10):
                looper = inf_mnist_loop(i)
                mnist_loopers.append(looper)

            while True:
                labels = np.random.randint(0, 5, size=batch_size)
                labels = torch.tensor(labels, dtype=torch.long)
                anti_labels = 9 - labels
                # x, y changed (anti-mnist 2)
                x = int_to_mnist(labels, mnist_loopers)
                y = int_to_mnist(anti_labels, mnist_loopers)
                yield x, y




