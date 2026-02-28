import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # First convolution
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c) # Normalize for stable training
        
        # Second convolution
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        
      # Non-linear activation
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)   # Learn features
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)  # Feature map
        p = self.pool(x)       # Downsampled map
        return x, p
    

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Upsampling: doubles height & width
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        
        # conv_block gets input = upsampled + skip features
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)            # Upsample
        x = torch.cat([x, skip], dim=1)  # Add skip connection
        x = self.conv(x)                # Learn details
        return x
    

class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        
        # Encoder: shrinking image, increasing channels
        self.e1 = encoder_block(in_ch, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)
        
        # Bottleneck: middle of the U
        self.b = conv_block(512, 1024)

        # Decoder: growing image back, using skip connections
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)
        
        # Final classifier: 1 output channel for binary segmentation
        self.output = nn.Conv2d(64, out_ch, kernel_size=1)  # 1 channel mask

    def forward(self, inputs):

        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Final prediction
        outputs = self.output(d4)
        return outputs
    

class ResNet50_UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, pretrained=True):
        super().__init__()

        if pretrained:
            resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            resnet = models.resnet50(weights=None)

        # ---------------- Encoder ----------------
        self.input_block = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )  # 64 ch

        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1  # 256 ch
        self.encoder2 = resnet.layer2  # 512 ch
        self.encoder3 = resnet.layer3  # 1024 ch
        self.encoder4 = resnet.layer4  # 2048 ch

        # ---------------- Bridge ----------------
        self.bridge = conv_block(2048, 2048)

        # ---------------- Skip Channel Alignment ----------------
        self.skip1_conv = nn.Conv2d(64, 64, kernel_size=1)
        self.skip2_conv = nn.Conv2d(256, 256, kernel_size=1)
        self.skip3_conv = nn.Conv2d(512, 512, kernel_size=1)
        self.skip4_conv = nn.Conv2d(1024, 1024, kernel_size=1)

        # ---------------- Decoder ----------------
        self.d1 = decoder_block(2048, 1024)
        self.d2 = decoder_block(1024, 512)
        self.d3 = decoder_block(512, 256)
        self.d4 = decoder_block(256, 64)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # ---------------- Output ----------------
        self.output = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):

        # -------- Encoder --------
        x1 = self.input_block(x)     # 64 ch (/2)
        x2 = self.maxpool(x1)        # (/4)

        x3 = self.encoder1(x2)       # 256 ch
        x4 = self.encoder2(x3)       # 512 ch
        x5 = self.encoder3(x4)       # 1024 ch
        x6 = self.encoder4(x5)       # 2048 ch

        # -------- Bridge --------
        b = self.bridge(x6)

        # -------- Align skip channels --------
        s1 = self.skip1_conv(x1)
        s2 = self.skip2_conv(x3)
        s3 = self.skip3_conv(x4)
        s4 = self.skip4_conv(x5)

        # -------- Decoder --------
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        d5 = self.final_up(d4)

        return self.output(d5)