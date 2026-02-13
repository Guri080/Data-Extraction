import torch
import torch.nn as nn
import torchvision.models as models

class _ResNetEncoder(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
    
    def forward(self, x):
        # Save features at each stage for skip connections
        x0 = self.relu(self.bn1(self.conv1(x)))  # 64 channels
        x1 = self.layer1(self.maxpool(x0))        # 256 channels
        x2 = self.layer2(x1)                      # 512 channels
        x3 = self.layer3(x2)                      # 1024 channels
        x4 = self.layer4(x3)                      # 2048 channels
        
        return x0, x1, x2, x3, x4  # Return all intermediate features


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Upsampling blocks - going from deep to shallow
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(1024 + 1024, 1024, kernel_size=3, padding=1)
        
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        
        # Final output layer
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x0, x1, x2, x3, x4):
        # x4 is the deepest feature (2048 channels)
        
        # Upsample and concatenate with x3
        d4 = self.up4(x4)
        d4 = torch.cat([d4, x3], dim=1)  # Concatenate skip connection
        d4 = self.conv4(d4)
        
        # Upsample and concatenate with x2
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.conv3(d3)
        
        # Upsample and concatenate with x1
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.conv2(d2)
        
        # Upsample and concatenate with x0
        d1 = self.up1(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.conv1(d1)
        
        # Final heatmap
        out = self.final(d1)
        out = self.sigmoid(out)
        
        return out


class HeatmapModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self._encoder = ResNetEncoder(resnet)
        self._decoder = Decoder()
    
    def forward(self, x):
        # Extract features at multiple scales
        x0, x1, x2, x3, x4 = self.encoder(x)
        
        # Decode with skip connections
        heatmap = self.decoder(x0, x1, x2, x3, x4)
        
        return heatmap