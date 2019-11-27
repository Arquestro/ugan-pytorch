# coding: utf-8

import torch


def encoder_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.LeakyReLU(0.2),
    )
    return block


def decoder_block(in_channels, out_channels):
    block = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
        torch.nn.ReLU(),
    )
    return block


class GeneratorNet(torch.nn.Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.encoder_module_list = torch.nn.ModuleList(
            [
                encoder_block(3, 64),
                encoder_block(64, 128),
                encoder_block(128, 256),
                encoder_block(256, 512),
                encoder_block(512, 512),
                encoder_block(512, 512),
                encoder_block(512, 512),
            ]
        )
        # Every decoder block result layer then concatenated with encoder layer
        # This is the reason why 2nd to last blocks have these sizes, doubled outputs
        self.decoder_module_list = torch.nn.ModuleList(
            [
                decoder_block(512, 512),
                decoder_block(1024, 512),
                decoder_block(1024, 512),
                decoder_block(1024, 256),
                decoder_block(512, 128),
                decoder_block(256, 64),
                torch.nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            ]
        )

    def forward(self, x):
        encoded_layers = dict()
        for i, encoder_module in enumerate(self.encoder_module_list):
            x = encoder_module(x)
            encoded_layers[str(i)] = x
        for i, decoder_module in enumerate(self.decoder_module_list):
            x = decoder_module(x)
            encoded_layer_index = len(self.decoder_module_list) - i - 2
            if encoded_layer_index >= 0:
                x = torch.cat([x, encoded_layers[str(encoded_layer_index)]], dim=1)
        x = torch.tanh(x)
        return x


class DiscriminatorNet(torch.nn.Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 4, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 4, stride=2, padding=2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, 4, stride=1, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1, 1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.net(x)
        return x
