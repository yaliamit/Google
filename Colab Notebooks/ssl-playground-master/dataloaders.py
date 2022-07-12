import torch
from lightly.data import LightlyDataset

from transforms import *


def get_clf_dataloaders(batch_size=256, num_workers=8):
    cifar10_train = torchvision.datasets.CIFAR10("./data", download=True, train=True)
    cifar10_test = torchvision.datasets.CIFAR10("./data", download=True, train=False)
    dataset_clf_train = LightlyDataset.from_torch_dataset(cifar10_train, transform=cifar10_train_classifier_transforms)
    dataset_clf_test = LightlyDataset.from_torch_dataset(cifar10_test, transform=cifar10_test_transforms)

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
