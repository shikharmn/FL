# -*- coding: utf-8 -*-
import os

import torchvision.datasets as datasets
import torchvision.transforms as transforms


"""the entry for classification tasks."""


def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_mnist(root, split, transform, target_transform, download):
    is_train = split == "train"

    if is_train:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )

def get_dataset(
    conf,
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == "cifar10" or name == "cifar100":
        return _get_cifar(name, root, split, transform, target_transform, download)
    elif name == "mnist":
        return _get_mnist(root, split, transform, target_transform, download)
    else:
        raise NotImplementedError