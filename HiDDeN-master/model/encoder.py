import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from vit_pytorch import ViT


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: HiDDenConfiguration):
        super(Encoder, self).__init__()
        self.H = config.H
        self.W = config.W
        self.conv_channels = config.encoder_channels
        self.num_blocks = config.encoder_blocks

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)
        self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

        self.vit = ViT(image_size=(config.H, config.W),
                       patch_size=32,
                       num_classes= 128*128*self.conv_channels,
                       dim=1024,
                       depth=config.decoder_blocks // 2,
                       heads=16,
                       mlp_dim=2048,
                       dropout=0.1,
                       emb_dropout=0.1)

    def forward(self, image, message):
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly

        semantic_representation = self.vit(image)
        semantic_representation = semantic_representation.reshape(image.shape[0],self.conv_channels,128,128)
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)

        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        encoded_image = self.conv_layers(image)

        # concatenate expanded message and image
        concat = torch.cat([expanded_message, semantic_representation, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w
