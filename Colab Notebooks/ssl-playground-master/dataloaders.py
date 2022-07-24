import torch
from lightly.data import LightlyDataset

from transforms import *


def get_clf_dataloaders(dataset='cifar10', batch_size=512, num_workers=8):
    if dataset == 'cifar10':
        torchdataset_train = torchvision.datasets.CIFAR10("./data", download=True, train=True)
        torchdataset_test = torchvision.datasets.CIFAR10("./data", download=True, train=False)
        input_size = 32
    elif dataset == 'cifar100':
        torchdataset_train = torchvision.datasets.CIFAR100("./data", download=True, train=True)
        torchdataset_test = torchvision.datasets.CIFAR100("./data", download=True, train=False)
        input_size = 32
    elif dataset == 'stl10':
        torchdataset_train = torchvision.datasets.STL10("./data", download=True, split='train')
        torchdataset_test = torchvision.datasets.STL10("./data", download=True, split='test')
        input_size = 96

    dataset_clf_train = LightlyDataset.from_torch_dataset(torchdataset_train,
                                                          transform=cifar10_train_classifier_transforms(input_size))
    dataset_clf_test = LightlyDataset.from_torch_dataset(torchdataset_test,
                                                         transform=cifar10_test_transforms(input_size))

    dataloader_clf_train = torch.utils.data.DataLoader(
        dataset_clf_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True,
    )

    dataloader_clf_test = torch.utils.data.DataLoader(
        dataset_clf_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return dataloader_clf_train, dataloader_clf_test
