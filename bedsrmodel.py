import os
import math
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============================================================
# U-Net building blocks
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # padding biar size sama
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)

# ============================================================
# BE-Net: Background Estimation + Attention
# ============================================================

class BENet(nn.Module):
    """
    Input  : dokumen shadow [B,3,H,W]
    Output : bg_color [B,3]     (average background doc)
             att_map  [B,1,H,W] (shadow attention)
    """
    def __init__(self, in_ch=3, base_ch=16):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.enc2 = Down(base_ch, base_ch * 2)
        self.enc3 = Down(base_ch * 2, base_ch * 4)

        self.att_head = nn.Conv2d(base_ch * 4, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_bg = nn.Linear(base_ch * 4, 3)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        # attention di feature map deep
        att = torch.sigmoid(self.att_head(x3))
        att = F.interpolate(att, size=x.shape[2:], mode="bilinear", align_corners=False)

        # background color global
        g = self.gap(x3).view(x3.size(0), -1)
        bg = torch.sigmoid(self.fc_bg(g))  # [B,3] / 0-1

        return bg, att
    
    # ============================================================
# SR-Net: Shadow Removal (U-Net style)
# ============================================================

class SRNet(nn.Module):
    """
    Input : concat(shadow_img[3], bg_map[3], att[1]) -> [B,7,H,W]
    Output: dokumen bebas bayangan [B,3,H,W]
    """
    def __init__(self, in_ch=7, out_ch=3, base_ch=32):
        super().__init__()
        self.inc   = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)

        self.up1   = Up(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up2   = Up(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up3   = Up(base_ch * 2 + base_ch, base_ch)

        self.outc  = OutConv(base_ch, out_ch)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x,  x2)
        x = self.up3(x,  x1)

        out = self.outc(x)
        out = torch.sigmoid(out)
        return out

# ============================================================
# Full BEDSR-like model
# ============================================================

class BEDSRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.benet = BENet()
        self.srnet = SRNet()

    def forward(self, shadow_img):
        """
        shadow_img: [B,3,H,W]
        return:
          pred_img : [B,3,H,W]
          bg_color : [B,3]
          att_map  : [B,1,H,W]
        """
        bg_color, att_map = self.benet(shadow_img)

        B, _, H, W = shadow_img.shape
        bg_map = bg_color.view(B, 3, 1, 1).expand(-1, -1, H, W)

        x_sr = torch.cat([shadow_img, bg_map, att_map], dim=1)  # [B,7,H,W]
        pred = self.srnet(x_sr)

        return pred, bg_color, att_map
