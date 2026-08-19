"""
Reference Repo: https://github.com/facebookresearch/AudioMAE
"""

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
import audioldm2.latent_diffusion.modules.audiomae.models_vit as models_vit
import audioldm2.latent_diffusion.modules.audiomae.models_mae as models_mae

# model = mae_vit_base_patch16(in_chans=1, audio_exp=True, img_size=(1024, 128))


class PatchEmbed_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class AudioMAE(nn.Module):
    """Audio Masked Autoencoder (MAE) pre-trained and finetuned on AudioSet (for SoundCLIP)"""

    def __init__(
        self,
    ):
        super().__init__()
        model = models_vit.__dict__["vit_base_patch16"](
            num_classes=527,
            drop_path_rate=0.1,
            global_pool=True,
            mask_2d=True,
            use_custom_patch=False,
        )

        img_size = (1024, 128)
        emb_dim = 768

        model.patch_embed = PatchEmbed_new(
            img_size=img_size,
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=emb_dim,
            stride=16,
        )
        num_patches = model.patch_embed.num_patches
        # num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
        model.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # checkpoint_path = '/mnt/bn/data-xubo/project/Masked_AudioEncoder/checkpoint/finetuned.pth'
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # msg = model.load_state_dict(checkpoint['model'], strict=False)
        # print(f'Load AudioMAE from {checkpoint_path} / message: {msg}')

        self.model = model

    def forward(self, x, mask_t_prob=0.0, mask_f_prob=0.0):
        """
        x: mel fbank [Batch, 1, T, F]
        mask_t_prob: 'T masking ratio (percentage of removed patches).'
        mask_f_prob: 'F masking ratio (percentage of removed patches).'
        """
        return self.model(x=x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)


class Vanilla_AudioMAE(nn.Module):
    """Audio Masked Autoencoder (MAE) pre-trained on AudioSet (for AudioLDM2)"""

    def __init__(
        self,
    ):
        super().__init__()
        model = models_mae.__dict__["mae_vit_base_patch16"](
            in_chans=1, audio_exp=True, img_size=(1024, 128)
        )

        # checkpoint_path = '/mnt/bn/lqhaoheliu/exps/checkpoints/audiomae/pretrained.pth'
        # checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # msg = model.load_state_dict(checkpoint['model'], strict=False)

        # Skip the missing keys of decoder modules (not required)
        # print(f'Load AudioMAE from {checkpoint_path} / message: {msg}')

        self.model = model.eval()

    def forward(self, x, mask_ratio=0.0, no_mask=False, no_average=False):
        """
        x: mel fbank [Batch, 1, 1024 (T), 128 (F)]
        mask_ratio: 'masking ratio (percentage of removed patches).'
        """
        with torch.no_grad():
            # embed: [B, 513, 768] for mask_ratio=0.0
            if no_mask:
                if no_average:
                    raise RuntimeError("This function is deprecated")
                    embed = self.model.forward_encoder_no_random_mask_no_average(
                        x
                    )  # mask_ratio
                else:
                    embed = self.model.forward_encoder_no_mask(x)  # mask_ratio
            else:
                raise RuntimeError("This function is deprecated")
                embed, _, _, _ = self.model.forward_encoder(x, mask_ratio=mask_ratio)
        return embed


if __name__ == "__main__":
    model = Vanilla_AudioMAE().cuda()
    input = torch.randn(4, 1, 1024, 128).cuda()
    print("The first run")
    embed = model(input, mask_ratio=0.0, no_mask=True)
    print(embed)
    print("The second run")
    embed = model(input, mask_ratio=0.0)
    print(embed)
