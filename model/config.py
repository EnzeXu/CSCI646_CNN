import argparse


SETTINGS = {
    "CIFAR10-CNN": {
        "dataset": "CIFAR10",
        "num_class": 10,
        "epochs": 90,
        "init_lr": 0.001,
        "batch_size": 64,
        "num_layers": 3,
        "layer_list": ["Conv2d", "Conv2d", "Conv2d"],
        "activation_list": ["ReLU", "ReLU", "ReLU"],
        "layer_size_list": [3, 32, 64, 128],
        "kernel_size_list": [3, 3, 3],
        "padding_list": [0, 0, 0],
        "stride_list": [1, 1, 1],
        "dropout_list": [0.2, 0.3, 0.4],
        "num_fc_layers": 2,
        "fc_layer_size_list": [512, 128, 10],
        "img_width": 32,
        "lr_scheduler_step_size": 15,
        "optimizer": "Adam",
    },
    "CIFAR100-ResNet18": {
        "dataset": "CIFAR100",
        "num_class": 240,
        "epochs": 100,
        "init_lr": 0.01,
        "batch_size": 64,
        "num_layers": 1,
        "layer_list": ["ResNet18"],
        "num_fc_layers": 2,
        "fc_layer_size_list": [512, 256, 100],
        "img_width": 32,
        "lr_scheduler_step_size": 40,
        "optimizer": "SGD",
    },
}


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--prob", type=str, default="CIFAR10-CNN", help="'CIFAR10-CNN' or 'CIFAR100-ResNet18'")
        parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--init", type=int, default=1, help="initialization of weights")

        self.args = parser.parse_args()

        for attr_name in vars(self.args):
            setattr(self, attr_name, getattr(self.args, attr_name))

        self.settings = SETTINGS.get(self.args.prob)


