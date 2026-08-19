import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class PatchEmbed_org(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        y = x.flatten(2).transpose(1, 2)
        return y


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
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)  # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2)  # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2)  # 32, 768, 1212 -> 32, 1212, 768
        return x


class PatchEmbed3D_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self,
        video_size=(16, 224, 224),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=768,
        stride=(2, 16, 16),
    ):
        super().__init__()

        self.video_size = video_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        _, _, t, h, w = self.get_output_shape(video_size)  # n, emb_dim, h, w
        self.patch_thw = (t, h, w)
        self.num_patches = t * h * w

    def get_output_shape(self, video_size):
        # todo: don't be lazy..
        return self.proj(
            torch.randn(1, self.in_chans, video_size[0], video_size[1], video_size[2])
        ).shape

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)  # 32, 3, 16, 224, 224 -> 32, 768, 8, 14, 14
        x = x.flatten(2)  # 32, 768, 1568
        x = x.transpose(1, 2)  # 32, 768, 1568 -> 32, 1568, 768
        return x


if __name__ == "__main__":
    # patch_emb = PatchEmbed_new(img_size=224, patch_size=16, in_chans=1, embed_dim=64, stride=(16,16))
    # input = torch.rand(8,1,1024,128)
    # output = patch_emb(input)
    # print(output.shape) # (8,512,64)

    patch_emb = PatchEmbed3D_new(
        video_size=(6, 224, 224),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=768,
        stride=(2, 16, 16),
    )
    input = torch.rand(8, 3, 6, 224, 224)
    output = patch_emb(input)
    print(output.shape)  # (8,64)
