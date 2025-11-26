import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================
#  Building blocks: ConvBlock, DownBlock, UpBlock
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)
        p = self.pool(x)
        return x, p


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # handle mismatch size (karena pooling/upsampling)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            diff_y = skip.size(-2) - x.size(-2)
            diff_x = skip.size(-1) - x.size(-1)
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# ============================================================
#  BENet (Brightness & Attention Estimator)
# ============================================================

class BENet(nn.Module):
    """
    Input : shadow image [B,3,H,W] range [-1,1]
    Output:
      - bg   : [B,3]        (background color, masih di [-1,1])
      - attn : [B,1,h,w]    (attention map low-res, sigmoid [0,1])
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.enc1 = DownBlock(in_ch, base_ch)
        self.enc2 = DownBlock(base_ch, base_ch * 2)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # attention map (1 channel)
        self.attention_head = nn.Conv2d(base_ch * 8, 1, kernel_size=1)

        # background color (3 channel, global pooling)
        self.bg_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch * 8, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,3,H,W]
        _, p1 = self.enc1(x)
        _, p2 = self.enc2(p1)
        _, p3 = self.enc3(p2)
        b = self.bottleneck(p3)  # [B, C, h, w]

        # attention
        attn = self.attention_head(b)       # [B,1,h,w]
        attn = torch.sigmoid(attn)

        # background color
        bg = self.bg_fc(b)                  # [B,3,1,1]
        bg = bg.view(bg.size(0), 3)         # [B,3]

        return bg, attn


# ============================================================
#  SRNet (Shadow Removal U-Net)
# ============================================================

class SRNet(nn.Module):
    """
    Input : concat(shadow, attention_up, bg_map) -> [B,7,H,W]
    Output: non-shadow image [B,3,H,W] range [-1,1] (tanh)
    """

    def __init__(self, in_ch: int = 7, base_ch: int = 64):
        super().__init__()
        self.down1 = DownBlock(in_ch, base_ch)
        self.down2 = DownBlock(base_ch, base_ch * 2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4)
        self.down4 = DownBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = UpBlock(base_ch * 16, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, p1 = self.down1(x)
        x2, p2 = self.down2(p1)
        x3, p3 = self.down3(p2)
        x4, p4 = self.down4(p3)

        b = self.bottleneck(p4)

        u4 = self.up4(b, x4)
        u3 = self.up3(u4, x3)
        u2 = self.up2(u3, x2)
        u1 = self.up1(u2, x1)

        out = self.out_conv(u1)
        out = torch.tanh(out)  # [-1,1]
        return out


# ============================================================
#  BEDSRNet (wrapper: BENet + SRNet)
# ============================================================

class BEDSRNet(nn.Module):
    """
    Wrapper:
      - Input : shadow [B,3,H,W] (range [-1,1])
      - Output:
          nonshadow : [B,3,H,W] range [-1,1]
          bg        : [B,3]     (background color)
          attn_up   : [B,1,H,W] (attention map full-res)
    """

    def __init__(self):
        super().__init__()
        self.benet = BENet()
        self.srnet = SRNet()

    def forward(self, x: torch.Tensor):
        """
        x: [B,3,H,W], range [-1,1]
        return:
          pred     : [B,3,H,W], range [-1,1]
          bg_color : [B,3]
          attn_up  : [B,1,H,W]
        """
        bg, attn = self.benet(x)  # bg [B,3], attn [B,1,h,w]

        # upsample attn ke resolusi input
        attn_up = F.interpolate(
            attn,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # background color map full-res
        bg_map = bg.view(bg.size(0), 3, 1, 1)
        bg_map = bg_map.expand(-1, -1, x.size(2), x.size(3))

        # concat -> SRNet
        gen_input = torch.cat([x, attn_up, bg_map], dim=1)  # [B,7,H,W]
        nonshadow = self.srnet(gen_input)  # [-1,1]

        return nonshadow, bg, attn_up
