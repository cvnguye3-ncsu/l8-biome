# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from itertools import chain

import torch as tf
import torch.nn as nn
from torch import Tensor

import pytorch_lightning as L

from timm.models.vision_transformer import Block
from timm.layers import to_2tuple

import numpy as np

from importables.pytorch.model import ModelSkeleton

class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

def _convTranspose2dOutput(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """
    Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )

class ConvTransformerTokensToEmbeddingNeck(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    """

    def __init__(
        self,
        embed_dim: int,
        output_embed_dim: int,
        # num_frames: int = 1,
        Hp: int = 14,
        Wp: int = 14,
        drop_cls_token: bool = True,
    ):
        """

        Args:
            embed_dim (int): Input embedding dimension
            output_embed_dim (int): Output embedding dimension
            Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
            Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
            drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
        """
        super().__init__()
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = Hp
        self.W_out = Wp
        # self.num_frames = num_frames

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x):
        if self.drop_cls_token:
            x = x[:, 1:, :]
            
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp)

        x = self.fpn1(x)
        x = self.fpn2(x)

        x = x.reshape((-1, self.output_embed_dim, self.H_out, self.W_out))

        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: 3d tuple of grid size: t, h, w
    return:
    pos_embed: L, D
    """

    assert embed_dim % 16 == 0

    t_size, h_size, w_size = grid_size

    w_embed_dim = embed_dim // 16 * 6
    h_embed_dim = embed_dim // 16 * 6
    t_embed_dim = embed_dim // 16 * 4

    w_pos_embed = get_1d_sincos_pos_embed_from_grid(
        w_embed_dim, np.arange(w_size))
    h_pos_embed = get_1d_sincos_pos_embed_from_grid(
        h_embed_dim, np.arange(h_size))
    t_pos_embed = get_1d_sincos_pos_embed_from_grid(
        t_embed_dim, np.arange(t_size))

    w_pos_embed = np.tile(w_pos_embed, (t_size * h_size, 1))
    h_pos_embed = np.tile(np.repeat(h_pos_embed, w_size, axis=0), (t_size, 1))
    t_pos_embed = np.repeat(t_pos_embed, h_size * w_size, axis=0)

    pos_embed = np.concatenate((w_pos_embed, h_pos_embed, t_pos_embed), axis=1)

    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class PatchEmbed(nn.Module):
    """ Frames of 2D Images to Patch Embedding
    The 3D version of timm.models.vision_transformer.PatchEmbed
    """

    def __init__(
            self,
            resize_embedding,
            img_size=224,
            patch_size=16,
            num_frames=3,
            tubelet_size=1,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.grid_size = (num_frames // tubelet_size,
                          img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * \
            self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        
        self.resize_embedding = resize_embedding

        if not resize_embedding:
            self.proj = nn.Conv3d(in_chans, embed_dim,
                                  kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                                  stride=(tubelet_size, patch_size[0], patch_size[1]),
                                  bias=bias) 
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim,
                                  kernel_size=(patch_size[0], patch_size[1]),
                                  stride=(patch_size[0], patch_size[1]),
                                  bias=bias)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: tf.Tensor):
        # B, C, T, H, W = x.shape

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B,C,T,H,W -> B,C,L -> B,L,C
        x = self.norm(x)

        return x

IMG_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768

class MaskedAutoencoderViT(ModelSkeleton):
    def __init__(self,
                 resize_embedding: bool, tubelet_size=1,
                 depth=24, num_heads=16, class_count=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 freeze_encoder_layers=0):
        super().__init__(resize_embedding, class_count)

        # === MODEL PARAMETERS ===
        self.patch_ct = IMG_SIZE // PATCH_SIZE

        real_frames = 1 if self.resize_embedding else 3
        in_channels = 3 if self.resize_embedding else 6
        
        # if real_frames > 1:
        #     self.input_shape = (in_channels, real_frames, IMG_SIZE, IMG_SIZE)
        self.input_shape = (3, IMG_SIZE, IMG_SIZE)
            
        # === MAE ENCODER ===
        self.patch_embed = PatchEmbed(self.resize_embedding, IMG_SIZE, PATCH_SIZE, 
                                      real_frames, tubelet_size, in_channels, EMBED_DIM)
        self.cls_token = nn.Parameter(tf.zeros(1, 1, EMBED_DIM))
        self.pos_embed = nn.Parameter(tf.zeros(1, self.patch_embed.num_patches + 1, EMBED_DIM))

        # Freeze the patch embedding
        if freeze_encoder_layers > 0 and not self.resize_embedding:
            self.cls_token.requires_grad_(False)
            self.patch_embed.requires_grad_(False)
            self.pos_embed.requires_grad_(False)

        self.norm = norm_layer(EMBED_DIM)

        # Freeze last normalization
        if freeze_encoder_layers == depth:
            self.norm.requires_grad_(False)
        
        self.blocks = nn.ModuleList(
            [Block(EMBED_DIM, num_heads, mlp_ratio,
                   qkv_bias=True, norm_layer=norm_layer)
             for _ in range(depth)])
        
        self.encoder = nn.Sequential(*(self.blocks + [self.norm]))

        # Freeze the blocks
        for block in self.blocks[:freeze_encoder_layers]:
            block.requires_grad_(False)
                
        self.encoder_params = chain(
            self.patch_embed.parameters(),
            [self.cls_token],
            [self.pos_embed],
            self.encoder.parameters()
        )
        
        # === MAE DECODER ===
        self.neck = ConvTransformerTokensToEmbeddingNeck(
            embed_dim=real_frames*EMBED_DIM,
            output_embed_dim=EMBED_DIM,
            drop_cls_token=True,
            Hp=IMG_SIZE // PATCH_SIZE,
            Wp=IMG_SIZE // PATCH_SIZE,)
        
        self.decoder = nn.Conv2d(in_channels=EMBED_DIM, out_channels=class_count,
                      kernel_size=3, padding=1, padding_mode='replicate')     
        
        self.decoder_params = chain(
            self.neck.parameters(),
            self.decoder.parameters()
        )   

        self._initialize_weights()

    # --- SETUP ---
    def _initialize_weights(self):
        # initialization
        
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(
            tf.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        tf.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        tf.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            tf.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _image_preprocessing(self, imgs):
        # Reorder RGB -> BGR
        imgs = imgs[:, [2, 1, 0], :, :] # B,C,H,W
        
        if not self.resize_embedding:
            # Duplicate for NIR and SWIR
            imgs = imgs.repeat(1, 2, 1, 1) # B,2C,H,W

            # Add in time dimension
            imgs = imgs.unsqueeze(2)
            imgs = imgs.repeat(1, 1, 3, 1, 1) # B,2C,T,H,W

        return imgs

    # --- FORWARD ---
    def _forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = tf.cat((cls_tokens, x), dim=1)

        x = self.encoder(x)

        return x

    def _forward_decoder(self, feat: Tensor):
        # (B, num_frames * patch total + 1, embedding size)
        logit = self.neck(feat)
        logit = self.decoder(logit)

        return logit

    def forward(self, x):
        x = self._image_preprocessing(x)
        
        latent = self._forward_encoder(x)
        logits = self._forward_decoder(latent)

        return logits