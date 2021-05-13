from torch import nn
import torch

class attention(nn.Module):
    def __init__(self, channel):
        super(attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.conv = nn.Conv2d(2, 1, 3, padding=1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        x = x * y

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        z = torch.cat([avg_out, max_out], dim=1)
        z = self.conv(z)
        z = self.Sigmoid(z)

        return z * x



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


class unetblock(nn.Module):
    def __init__(self, num):
        super(unetblock, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        num = [num, num*2]

        self.conv1 = conv(32, 64)
        self.conv2 = conv(64, 128)
        self.conv3 = conv(128, 256)

        #self.res1 = conv(64, 64)
        #self.res2 = conv(128, 128)
        #self.res3 = conv(256, 256)
        #self.res4 = conv(512, 512)
        
        #self.conv4 = conv(768, 768)
        #self.conv5 = conv(640, 640)
        #self.conv6 = conv(576, 576)
        
        self.res5 = conv(384, 256)
        self.res6 = conv(320, 128)
        self.res7 = conv(160, 32)
        #self.res8 = conv(128, 32)

        self.attention1 = attention(256)
        self.attention2 = attention(128)
        self.attention3 = attention(32)
        



    def forward(self, input):
        x1_1 = input
        x2_1 = self.conv1(self.pool(x1_1))
        x3_1 = self.conv2(self.pool(x2_1))
        x4_1 = self.conv3(self.pool(x3_1))
        

        
        x3_3 = self.res5(torch.cat([x3_1, self.up(x4_1)], 1))
        x3_3 = self.attention1(x3_3)
        
        x2_3 = self.res6(torch.cat([x2_1, self.up(x3_3)], 1))
        x2_3 = self.attention2(x2_3)
        
        x1_3 = torch.cat([x1_1, self.up(x2_3)], 1)
        x_out = self.attention3(self.res7(x1_3))
        
        return x_out


class dehazenet(nn.Module):
    def __init__(self):
        super(dehazenet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)

        self.ublock1 = unetblock(1)
        self.ublock2 = unetblock(2)

        self.attention = attention(64)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):

         out = self.conv1(x)
         out1 = self.ublock1(out)
         out2 = self.ublock2(out1)
         out2 = self.conv3(self.conv2(self.attention(torch.cat([out1, out2], dim=1))))

         return out2 + x






