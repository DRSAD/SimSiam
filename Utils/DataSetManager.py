from torchvision import transforms
import torchvision
from Utils.SimSiamAugmentations import SimSiamTransform


def get_dataset(dataset_name,data_dir):
    train_dataset,test_dataset=None,None
    if dataset_name == 'CIFAR10_SSL':
        train_transform = SimSiamTransform(32,mean=[0.49, 0.48, 0.45],std=[0.25, 0.24, 0.26])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.49, 0.48, 0.45],std=[0.25, 0.24, 0.26])
                                             ])

        train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        test_dataset= torchvision.datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    if dataset_name == 'CIFAR10':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.49, 0.48, 0.45],std=[0.25, 0.24, 0.26])
        ])
        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.49, 0.48, 0.45], std=[0.25, 0.24, 0.26])])

        train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, transform=test_transform, download=True)


    return train_dataset,test_dataset






