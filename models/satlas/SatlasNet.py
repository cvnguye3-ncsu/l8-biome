import collections
from itertools import chain

import torch
import torch.nn
import torchvision

import pytorch_lightning as L

from importables.pytorch.model import ModelSkeleton

def adjust_state_dict_prefix(state_dict, needed, prefix=None, prefix_allowed_count=None):
    """
    Adjusts the keys in the state dictionary by replacing 'backbone.backbone' prefix with 'backbone'.

    Args:
        state_dict (dict): Original state dictionary with 'backbone.backbone' prefixes.

    Returns:
        dict: Modified state dictionary with corrected prefixes.
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        # Assure we're only keeping keys that we need for the current model component. 
        if not needed in key:
            continue

        # Update the key prefixes to match what the model expects.
        if prefix is not None:
            while key.count(prefix) > prefix_allowed_count:
                key = key.replace(prefix, '', 1)

        new_state_dict[key] = value
    return new_state_dict

class SimpleHead(torch.nn.Module):
    def __init__(self, backbone_channels, num_categories=2):
        super(SimpleHead, self).__init__()

        use_channels = backbone_channels[0][1]
        num_layers = 2
        self.num_outputs = num_categories

        layers = []
        for _ in range(num_layers - 1):
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(use_channels, use_channels, 3, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)

        # bin_segment
        layers.append(torch.nn.Conv2d(use_channels, self.num_outputs, 3, padding=1))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, raw_features: list[torch.Tensor]) -> torch.Tensor:
        raw_outputs = self.layers(raw_features[0])
        
        # bin_segment
        outputs = torch.nn.functional.softmax(raw_outputs, dim=1)

        return outputs

class FPN(torch.nn.Module):
    def __init__(self, backbone_channels):
        super(FPN, self).__init__()

        out_channels = 128
        in_channels_list = [ch[1] for ch in backbone_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)

        self.out_channels = [[ch[0], out_channels] for ch in backbone_channels]

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(x)])
        output = self.fpn(inp)
        output = list(output.values())

        return output

class Upsample(torch.nn.Module):
    # Computes an output feature map at 1x the input resolution.
    # It just applies a series of transpose convolution layers on the
    # highest resolution features from the backbone (FPN should be applied first).

    def __init__(self, backbone_channels):
        super(Upsample, self).__init__()
        self.in_channels = backbone_channels

        out_channels = backbone_channels[0][1]
        self.out_channels = [(1, out_channels)] + backbone_channels

        layers = []
        depth, ch = backbone_channels[0]
        while depth > 1:
            next_ch = max(ch//2, out_channels)
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(ch, ch, 3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1),
                torch.nn.ReLU(inplace=True),
            )
            layers.append(layer)
            ch = next_ch
            depth /= 2

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        output = self.layers(x[0])
        
        return [output] + x

class SwinBackbone(torch.nn.Module):
    def __init__(self, num_channels):
        super(SwinBackbone, self).__init__()

        self.backbone = torchvision.models.swin_v2_b()
        self.out_channels = [
            [4, 128],
            [8, 256],
            [16, 512],
            [32, 1024],
        ]

        self.backbone.features[0][0] = torch.nn.Conv2d(num_channels, 
                                                       self.backbone.features[0][0].out_channels, 
                                                       kernel_size=(4, 4), stride=(4, 4))

    def forward(self, x):
        outputs = []
        
        # Skip over the norm and head of original model.
        for layer in self.backbone.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
            
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]

class Model(ModelSkeleton):
    def __init__(self, 
                 num_categories, fpn=False, weights=None, 
                 freeze_encoder_layers: int = 12, resize_embedding = False):
        super().__init__(resize_embedding, num_categories)
        
        # === MODEL PARAMETERS ===
        self.fpn_flag = fpn
        num_channels = 11 
        
        # === ENCODER ===
        self.backbone = SwinBackbone(num_channels)

        # Load pretrained weights into the intialized backbone if weights were specified.
        if weights is not None:
            state_dict = adjust_state_dict_prefix(weights, 'backbone', 'backbone.', 1)
            self.backbone.load_state_dict(state_dict)
            
        if resize_embedding:
            num_channels = 3
            self.backbone.backbone.features[0][0] = torch.nn.Conv2d(num_channels, 
                                           self.backbone.backbone.features[0][0].out_channels, 
                                            kernel_size=(4, 4), stride=(4, 4))
            
        self.input_shape = (num_channels, 224, 224)
        
        for idx in [0, 2, 4, 6][:freeze_encoder_layers]:
            self.backbone.backbone.features[idx].requires_grad_(False)
            self.backbone.backbone.features[idx + 1].requires_grad_(False)
            
        self.encoder_params = self.backbone.backbone.features.parameters()
        
        # === DECODER ===
        self.fpn = FPN(self.backbone.out_channels)
        self.upsample = Upsample(self.fpn.out_channels if self.fpn_flag else self.backbone.out_channels)
        self.head = SimpleHead(self.fpn.out_channels if self.fpn_flag else self.backbone.out_channels, num_categories)
        
        self.decoder_params = chain(self.upsample.parameters(),
                                    self.head.parameters())
        
        if self.fpn_flag:
            self.decoder_params = chain(self.decoder_params, self.fpn.parameters())

    def _image_preprocessing(self, x) -> torch.Tensor:
        x = x[:, [2, 1, 0], :, :]
        
        if not self.resize_embedding:
            extra_chans = x[:, :2, :, :]
            x = torch.cat((x, x, x, extra_chans), dim=1)
            
        return x

    def forward(self, imgs):
        x = self._image_preprocessing(imgs)
        
        # Define forward pass
        x = self.backbone(x)
        
        x = self.fpn(x)
            
        x = self.upsample(x)
        x = self.head(x)
        
        return x