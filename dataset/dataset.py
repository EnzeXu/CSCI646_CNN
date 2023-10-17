import torchvision
import os
from torchvision import transforms
from torch.utils.data import DataLoader


def get_dataset(config):
    data_folder_path = "./data/"
    if not os.path.exists(data_folder_path):
        os.makedirs(data_folder_path)


    assert config.settings["dataset"] in ["CIFAR10", "CIFAR100"]
    if config.settings["dataset"] == "CIFAR10":
        train_trans = transforms.Compose([transforms.RandomRotation(degrees=(-10, 10)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = torchvision.datasets.CIFAR10(root=data_folder_path, download=True, train=True,
                                                     transform=train_trans)
        test_dataset = torchvision.datasets.CIFAR10(root=data_folder_path, download=False, train=False,
                                                    transform=test_trans)
    else:
        train_trans = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        test_trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        train_dataset = torchvision.datasets.CIFAR100(root=data_folder_path, download=True, train=True,
                                                     transform=train_trans)
        test_dataset = torchvision.datasets.CIFAR100(root=data_folder_path, download=False, train=False,
                                                    transform=test_trans)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.settings["batch_size"], shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.settings["batch_size"], shuffle=False, num_workers=8)

    print(f"[Dataset size] Train: {len(train_loader)}, Test: {len(test_loader)}, Total: {len(train_loader) + len(test_loader)}")

    return train_loader, test_loader