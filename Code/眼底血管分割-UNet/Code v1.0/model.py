import torch
from torch import nn
from torch.nn import functional as F

# 共性的卷积层
class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvLayer, self).__init__()
        # 使用序列构造器开始构造
        self.layer = nn.Sequential(
            # reflect:使用特征镜像做padding，可以增加特征值
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 下采样改用卷积来取代max-pooling，因为max-pooling没有特征提取的能力，而且丢特征丢的太多了
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


# 最邻近插值法进行上采样，除此之外还需使用1x1的卷积降通道
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel//2, kernel_size=1, stride=1)     # 1x1的卷积仅用于降通道，而不会特征提取

    def forward(self, x, left_out):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((left_out, out), dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # contract path
        self.conv1 = ConvLayer(3, 64)
        self.down1 = DownSample(64)
        self.conv2 = ConvLayer(64, 128)
        self.down2 = DownSample(128)
        self.conv3 = ConvLayer(128, 256)
        self.down3 = DownSample(256)
        self.conv4 = ConvLayer(256, 512)
        self.down4 = DownSample(512)
        self.conv5 = ConvLayer(512, 1024)
        # expansive path
        self.up1 = UpSample(1024)
        self.conv6 = ConvLayer(1024, 512)
        self.up2 = UpSample(512)
        self.conv7 = ConvLayer(512, 256)
        self.up3 = UpSample(256)
        self.conv8 = ConvLayer(256, 128)
        self.up4 = UpSample(128)
        self.conv9 = ConvLayer(128, 64)

        self.out = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.active_func = nn.Sigmoid()

    def forward(self, x):
        r1 = self.conv1(x)
        r2 = self.conv2(self.down1(r1))
        r3 = self.conv3(self.down2(r2))
        r4 = self.conv4(self.down3(r3))
        r5 = self.conv5(self.down4(r4))
        o1 = self.conv6(self.up1(r5, r4))
        o2 = self.conv7(self.up2(o1, r3))
        o3 = self.conv8(self.up3(o2, r2))
        o4 = self.conv9(self.up4(o3, r1))

        return self.active_func(self.out(o4))


# 验证一下网络结构

if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    model = UNet()
    print(model(x).shape)



"""
# contracting path(左侧)的每一层架构都相同
class downSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downSampleLayer, self).__init__()  # downSample继承nn.Module，先进行父类初始化
        # 2次卷积+BatchNorm+ReLU
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # 下采样
        # 用卷积做下采样
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out_copy = self.conv_bn_relu(x)     # 用于拼接到右侧
        out_down = self.downsample(out_copy)     # 用于向更深层传递
        return out_copy, out_down


# expansive path(右侧)的每一层架构都相同(除了最后一层)
class upSampleLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upSampleLayer, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2, out_channels=out_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, out_copy):
        out_up = self.conv_bn_relu(x)
        out_up = self.upsample(out_up)
        out = torch.cat((out_copy, out_up), dim=1)  # 拼接
        return out


# 搭建U-Net网络
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        channels = [2**(i+6) for i in range(5)]  # [64, 128, 256, 512, 1024]
        # Contracting Path
        self.d1 = downSampleLayer(3, channels[0]) # 3-64
        self.d2 = downSampleLayer(channels[0], channels[1])    # 64-128
        self.d3 = downSampleLayer(channels[1], channels[2])    # 128-256
        self.d4 = downSampleLayer(channels[2], channels[3])    # 256-512
        # Expansive Path
        self.u1 = upSampleLayer(channels[3], channels[3])   # 512-1024-512
        self.u2 = upSampleLayer(channels[4], channels[2])   # 1024-512-256
        self.u3 = upSampleLayer(channels[3], channels[1])   # 512-256-128
        self.u4 = upSampleLayer(channels[2], channels[0])   # 256-128-64
        # Output
        self.o=nn.Sequential(
            nn.Conv2d(channels[1], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out_1, out1 = self.d1(x)
        out_2, out2 = self.d2(out1)
        out_3, out3 = self.d3(out2)
        out_4, out4 = self.d4(out3)
        out5 = self.u1(out4, out_4)
        out6 = self.u2(out5, out_3)
        out7 = self.u3(out6, out_2)
        out8 = self.u4(out7, out_1)
        out = self.o(out8)
        return out
        
"""