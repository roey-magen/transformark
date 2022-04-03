import torch
import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu
from vit_pytorch import ViT
from transformers import ViTFeatureExtractor, ViTModel



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
        self.encoder_mode = config.encoder_mode

        layers = [ConvBNRelu(3, self.conv_channels)]

        for _ in range(config.encoder_blocks-1):
            layer = ConvBNRelu(self.conv_channels, self.conv_channels)
            layers.append(layer)

        self.conv_layers = nn.Sequential(*layers)

        self.final_layer = nn.Conv2d(self.conv_channels, 3, kernel_size=1)

        if self.encoder_mode == 'vit': # Attempt 1 in the report
            self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                             self.conv_channels)
            self.vit = ViT(image_size=(config.H, config.W),
                        patch_size=32,
                        num_classes= 128*128*self.conv_channels,
                        dim=1024,
                        depth=config.decoder_blocks // 2,
                        heads=16,
                        mlp_dim=2048,
                        dropout=0.1,
                        emb_dropout=0.1)

        elif self.encoder_mode == 'dino-output': # Attempt 2 in the report
            self.after_concat_layer = ConvBNRelu(self.conv_channels + 3 + config.message_length,
                                        self.conv_channels)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
            self.model = ViTModel.from_pretrained('facebook/dino-vits8')
            self.model.eval()
            self.linear = nn.Linear(in_features=384, out_features=128*128*self.conv_channels)

        elif self.encoder_mode == 'dino-attention': # Attempt 3 in the report
            self.after_concat_layer = ConvBNRelu(39, self.conv_channels)
            self.feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits8')
            self.model = ViTModel.from_pretrained('facebook/dino-vits8', output_attentions=True)
            self.model.eval()
            
        else: 
            raise Exception('encoder_mode is illegal. The value of encoder_mode was: {}'.format(self.encoder_mode))


    
    def forward_vit(self, image, message):
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

    def forward_dino_output(self, image, message):
        image_list = [x.squeeze(0) for x in torch.split(image.cpu(), 1)]
        inputs = self.feature_extractor(image_list, return_tensors="pt").to('cuda')
        outputs = self.model(**inputs)
        semantic_representation = outputs.pooler_output
        semantic_representation = self.linear(semantic_representation)

        semantic_representation = semantic_representation.reshape(image.shape[0],self.conv_channels,128,128)
      
        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)

        concat = torch.cat([expanded_message, semantic_representation, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w

    
    def forward_dino_attention(self, image, message):
        image_list = [x.squeeze(0) for x in torch.split(image.cpu(), 1)]
        inputs = self.feature_extractor(image_list, return_tensors="pt").to('cuda')
        outputs = self.model(**inputs)
        attention = outputs.attentions[-1][:,:,0,1:]
        attention = attention.reshape(attention.shape[0], attention.shape[1], 28, 28)
        attention = nn.functional.interpolate(attention, 128)

        # First, add two dummy dimensions in the end of the message.
        # This is required for the .expand to work correctly
        expanded_message = message.unsqueeze(-1)
        expanded_message.unsqueeze_(-1)
        expanded_message = expanded_message.expand(-1,-1, self.H, self.W)
        concat = torch.cat([expanded_message, attention, image], dim=1)
        im_w = self.after_concat_layer(concat)
        im_w = self.final_layer(im_w)
        return im_w

    def forward(self, image, message):
        if self.encoder_mode == 'vit':
            return forward_vit(self, image, message)
        elif self.encoder_mode == 'dino-output':
            return forward_dino_output(self, image, message)
        elif self.encoder_mode == 'dino-attention':
            return forward_dino_attention(self, image, message)
        else: 
            raise Exception('encoder_mode is illegal. The value of encoder_mode was: {}'.format(self.encoder_mode))
