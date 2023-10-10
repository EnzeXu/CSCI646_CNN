import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.conv = nn.Sequential(  # nn.Conv2d(in_channels,...),
            # activation fun,
            # dropout,
            # nn.Conv2d(in_channels,...),
            # activation fun,
            # dropout,
            ## continue like above,
            ## **define pooling (bonus)**,
        )

        self.config = config

        self.num_layers = config.settings["num_layers"]
        self.num_fc_layers = config.settings["num_fc_layers"]
        self.layers = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_layers = nn.ModuleList()

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
                activation = getattr(nn, config.settings["activation_list"][i])()
                dropout = nn.Dropout(config.settings["dropout_list"][i])
                batch_norm = nn.BatchNorm2d(out_channels)

                self.layers.append(conv_layer)
                self.layers.append(activation)
                self.layers.append(batch_norm)
                self.layers.append(dropout)

        for i in range(self.num_fc_layers):
            in_channels = config.settings["fc_layer_size_list"][i]
            out_channels = config.settings["fc_layer_size_list"][i + 1]
            self.fc_layers.append(nn.Linear(
                in_features=in_channels,
                out_features=out_channels,
            ))


        ##------------------------------------------------
        ## write code to define fully connected layer below
        ##------------------------------------------------
        # in_size =
        # out_size =
        # self.fc = nn.Linear(in_size, out_size)

    '''feed features to the model'''

    def forward(self, x):  # default
        for layer in self.layers:
            x = layer(x)

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        # print("x shape:", x.shape)
        for layer in self.fc_layers:
            x = layer(x)
        # x = self.fc_layer(x)

        return x

if __name__ == "__main__":
    from config import Config

    model = Model(Config())
    print(model)