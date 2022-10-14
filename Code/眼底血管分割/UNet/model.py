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
        self.layer = nn.Conv2d(channel, channel // 2, kernel_size=1, stride=1)  # 1x1的卷积仅用于降通道，而不会特征提取

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

        # self.out = nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
        self.out = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
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
    x = torch.randn(2, 3, 576, 576)
    model = UNet()
    print(model(x).shape)
