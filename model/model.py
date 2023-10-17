import torch
import torchvision
import torch.nn as nn


class ModelCNN(nn.Module):
    def __init__(self, config):
        super(ModelCNN, self).__init__()

        self.config = config

        self.num_layers = config.settings["num_layers"]
        self.num_fc_layers = config.settings["num_fc_layers"]
        self.layers = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_layers = nn.ModuleList()

        self.resnet18 = None
        if "ResNet18" in config.settings["layer_list"]:
            self.resnet18 = torchvision.models.resnet18(pretrained=True)
            self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])

        for i in range(self.num_layers):
            if config.settings["layer_list"][i] == "Conv2d":
                in_channels = config.settings["layer_size_list"][i]
                out_channels = config.settings["layer_size_list"][i + 1]
                kernel_size = config.settings["kernel_size_list"][i]
                padding = config.settings["padding_list"][i]
                stride = config.settings["stride_list"][i]
                conv_layer = getattr(nn, config.settings["layer_list"][i])(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride)
                if self.config.init:
                    nn.init.xavier_uniform_(conv_layer.weight)
                activation = getattr(nn, config.settings["activation_list"][i])()
                dropout = nn.Dropout(config.settings["dropout_list"][i])
                batch_norm = nn.BatchNorm2d(out_channels)
                pooling = self.pooling

                self.layers.append(conv_layer)
                self.layers.append(batch_norm)
                self.layers.append(activation)
                self.layers.append(pooling)
                self.layers.append(dropout)
            elif config.settings["layer_list"][i] == "ResNet18":
                self.layers.append(self.resnet18)

        for i in range(self.num_fc_layers):
            in_channels = config.settings["fc_layer_size_list"][i]
            out_channels = config.settings["fc_layer_size_list"][i + 1]
            linear_layer = nn.Linear(
                in_features=in_channels,
                out_features=out_channels,
            )
            if self.config.init:
                nn.init.xavier_uniform_(linear_layer.weight)
            self.fc_layers.append(linear_layer)
            if i != self.num_fc_layers - 1:
                self.fc_layers.append(nn.BatchNorm1d(out_channels))
                self.fc_layers.append(nn.ReLU())
                self.fc_layers.append(nn.Dropout(0.2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x


class ModelResNet18(nn.Module):
    def __init__(self, config):
        super(ModelResNet18, self).__init__()

        self.config = config

        self.num_class = config.settings["num_class"]

        # self.resnet18 = torchvision.models.resnet18(pretrained=False)
        # self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 256)
        #
        # self.fc_layers = nn.ModuleList()
        # self.fc_layers.append(nn.BatchNorm1d(256))
        # self.fc_layers.append(nn.ReLU())
        # self.fc_layers.append(nn.Dropout(0.2))
        # self.fc_layers.append(nn.Linear(256, 100))

        self.conv_layer_1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv_layer_2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.res_layer1 = torch.nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_layer_3 = torch.nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer_4 = torch.nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

        self.res_layer2 = torch.nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(512 * 2 * 2,  256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 100),
        )

    def forward(self, x):
        x = self.resnet18(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    from config import Config

    model = ModelCNN(Config())
    print(model)