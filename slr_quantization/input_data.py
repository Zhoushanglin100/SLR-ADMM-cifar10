import torch
from torchvision import datasets, transforms


class MNISTDataLoader:
    def __init__(self, batch_size, test_batch_size, kwargs):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

# -----------------------------------------------------------------------------------
class CIFAR10DataLoader:
    def __init__(self, batch_size, test_batch_size, kwargs):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])

        self.train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, download=True, 
                         transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, download=True, 
                             transform=transforms.Compose([
                                    # transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    normalize,
                                ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)
