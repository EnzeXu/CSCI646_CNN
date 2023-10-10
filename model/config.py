import argparse

SETTINGS = {
    "CIFAR10-CNN": {
        "epochs": 90,
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
    },
}


class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--prob", type=str, default="CIFAR10-CNN", help="'CIFAR10-CNN' or 'CIFAR100-ResNet18'")
        # parser.add_argument("--epoch", type=int, default=50, help="Number of epoch")
        parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")
        parser.add_argument("--seed", type=int, default=0, help="Random seed")
        parser.add_argument("--init", type=int, default=1, help="initialization of weights")

        # parser.add_argument("-fc_hidden1", dest="fc_hidden1", type=int, default=100, help="dim of hidden neurons")
        # parser.add_argument("-fc_hidden2", dest="fc_hidden2", type=int, default=100, help="dim of hidden neurons")
        parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (base, with adjust lr func)")
        # parser.add_argument("-decay", dest="decay", type=float, default=0.5, help="learning rate")
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        # parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
        # parser.add_argument("-rotation", dest="rotation", type=int, default=10, help="image rotation")
        # parser.add_argument("-load_checkpoint", dest="load_checkpoint", type=str2bool, default=True,
        #                     help="true of false")

        parser.add_argument("--activation", dest="activation", type=str, default="relu", help="Activation function from ['relu', 'tanh', 'elu', 'selu', 'sigmoid'] ")
        # parser.add_argument("-MC", dest='MC', type=int, default=10, help="number of monte carlo")
        # parser.add_argument("-channel_out1", dest='channel_out1', type=int, default=64, help="number of channels")
        # parser.add_argument("-channel_out2", dest='channel_out2', type=int, default=64, help="number of channels")
        # parser.add_argument("-k_size", dest='k_size', type=int, default=4, help="size of filter")
        # parser.add_argument("-pooling_size", dest='pooling_size', type=int, default=2, help="size for max pooling")
        # parser.add_argument("-stride", dest='stride', type=int, default=1, help="stride for filter")
        # parser.add_argument("-max_stride", dest='max_stride', type=int, default=2, help="stride for max pooling")
        # parser.add_argument("-ckp_path", dest='ckp_path', type=str, default="checkpoint", help="path of checkpoint")

        self.args = parser.parse_args()

        for attr_name in vars(self.args):
            setattr(self, attr_name, getattr(self.args, attr_name))

        self.settings = SETTINGS.get(self.args.prob)
