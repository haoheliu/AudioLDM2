# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self, global_pool=False, mask_2d=True, use_custom_patch=False, **kwargs
    ):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
        del self.norm  # remove the original norm
        self.mask_2d = mask_2d
        self.use_custom_patch = use_custom_patch

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:
            # # for AS
            T = 101  # 64,101
            F = 12  # 8,12
            # # for ESC
            # T=50
            # F=12
            # for SPC
            # T=12
            # F=12
        else:
            # ## for AS
            T = 64
            F = 8
            # ## for ESC
            # T=32
            # F=8
            ## for SPC
            # T=8
            # F=8

        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        # x_masked = torch.gather(x, dim=1, index=index)
        # x_masked = x_masked.reshape(N,len_keep_T*F,D)
        x = torch.gather(x, dim=1, index=index)  # N, len_keep_T(T'), F, D

        # mask F
        # x = x.reshape(N, T, F, D)
        x = x.permute(0, 2, 1, 3)  # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        # index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3)  # N F' T' D => N T' F' D
        # x_masked = x_masked.reshape(N,len_keep*T,D)
        x_masked = x_masked.reshape(N, len_keep_F * len_keep_T, D)

        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0]  # 4,1,1024,128
        x = self.patch_embed(x)  # 4, 512, 768

        x = x + self.pos_embed[:, 1:, :]
        if self.random_masking_2d:
            x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob, mask_f_prob)
        else:
            x, mask, ids_restore = self.random_masking(x, mask_t_prob)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    # overwrite original timm
    def forward(self, x, v=None, mask_t_prob=0.0, mask_f_prob=0.0):
        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x = self.forward_features_mask(
                x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob
            )
        else:
            x = self.forward_features(x)
        x = self.head(x)
        return x


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model
