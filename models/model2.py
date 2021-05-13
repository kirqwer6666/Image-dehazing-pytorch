from torch import nn
import torch


class attention(nn.Module):
    def __init__(self, channel):
        super(attention, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv(x)

        return x


class conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):

        out = self.conv1(x)
        out += x

        return self.conv2(out)


class unetpp(nn.Module):
    def __init__(self, channels):
        super(unetpp, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        nb_filter = [channels, channels*2, channels*4, channels*8]

        #self.conv0_0 = conv(in_channel, nb_filter[0])
        self.conv1_0 = conv(nb_filter[0], nb_filter[1])
        self.conv2_0 = conv(nb_filter[1], nb_filter[2])
        self.conv3_0 = conv(nb_filter[2], nb_filter[3])

        self.conv0_1 = conv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = conv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = conv(nb_filter[2]+nb_filter[3], nb_filter[2])

        self.conv0_2 = conv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = conv(nb_filter[1]*2+nb_filter[2], nb_filter[1])

        self.conv0_3 = conv(nb_filter[0]*3+nb_filter[1], nb_filter[0])

        self.atten0_1 = attention(nb_filter[0])
        self.atten1_1 = attention(nb_filter[1])
        self.atten0_2 = attention(nb_filter[0])
        self.atten2_1 = attention(nb_filter[2])
        self.atten1_2 = attention(nb_filter[1])
        self.atten0_3 = attention(nb_filter[0])

    def forward(self, input):
        x0_0 = input
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.atten0_1(self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1)))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.atten1_1(self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1)))
        x0_2 = self.atten0_2(self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1)))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.atten2_1(self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1)))
        x1_2 = self.atten1_2(self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1)))
        x0_3 = self.atten0_3(self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1)))

        return x0_3

class noattention(nn.Module):
    def __init__(self):
        super(noattention, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)

        self.unetpp1 = unetpp(32)
        self.unetpp2 = unetpp(32)

        self.atten = attention(64)
        self.conv2 = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):

         out = self.conv1(x)
         out1 = self.unetpp1(out)
         out2 = self.unetpp2(out1)
         out2 = self.conv2(self.atten(torch.cat([out1, out2], dim=1)))

         return out2 + x






