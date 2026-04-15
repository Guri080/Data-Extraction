import torch
import torch.nn as nn
import torchvision.models as models


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )


class _ResNetEncoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))  # 112×112×64
        x1 = self.layer1(self.maxpool(x0))        # 56×56×256
        x2 = self.layer2(x1)                      # 28×28×512
        x3 = self.layer3(x2)                      # 14×14×1024
        x4 = self.layer4(x3)                      # 7×7×2048
        return x0, x1, x2, x3, x4


class _Decoder(nn.Module):
    def __init__(self, num_keypoints=1):
        super().__init__()

        self.up4   = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv4 = conv_block(1024 + 1024, 1024)

        self.up3   = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = conv_block(512 + 512, 512)

        self.up2   = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = conv_block(256 + 256, 256)

        self.up1   = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv1 = conv_block(64 + 64, 64)

        # no skip connection available at 224×224, just upsample and refine
        self.up0   = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv0 = conv_block(64, 64)

        self.final = nn.Sequential(
            nn.Conv2d(64, num_keypoints, kernel_size=1),
        )

    def forward(self, x0, x1, x2, x3, x4):
        d4 = self.conv4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.conv3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.conv2(torch.cat([self.up2(d3), x1], dim=1))
        d1 = self.conv1(torch.cat([self.up1(d2), x0], dim=1))
        d0 = self.conv0(self.up0(d1))
        return self.final(d0)


class HeatmapModel(nn.Module):
    def __init__(self, num_keypoints=1, freeze_encoder=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = _ResNetEncoder(resnet)
        self.decoder = _Decoder(num_keypoints=num_keypoints)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        return self.decoder(x0, x1, x2, x3, x4)