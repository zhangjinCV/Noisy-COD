import torch.nn as nn
import torch
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b4


class Network(nn.Module):
    def __init__(self, channels=64):
        super(Network, self).__init__()
        self.shared_encoder = pvt_v2_b4()
        self.shared_encoder.load_state_dict(torch.load('/mnt/jixie16t/dataset/imagenet_pretrained_weight/PVTv2/pvt_v2_b4.pth'))
        self.GCM3 = GCM3([64, 128, 320, 512], channels)

        self.LL_down3 = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )

        self.LL_down4 = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )

        self.dePixelShuffle = torch.nn.Upsample(scale_factor=2)
        self.one_conv_f4_ll = ETM(channels * 2, channels)
        self.one_conv_f3_ll = ETM(channels * 2, channels)

        self.one_conv_f1_hh = ETM(channels * 2, channels)
        self.one_conv_f2_hh = ETM(channels * 2, channels)

        self.GPM = GPM(512)
        self.decoder = UNetDecoderWithEdges(channels, channels)

    def forward(self, x):
        en_feats = self.shared_encoder(x)
        x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        HH_up = self.dePixelShuffle(HH)
        f1_HH = torch.cat([HH_up, f1], dim=1)
        f1_HH = self.one_conv_f1_hh(f1_HH)

        f2_HH = torch.cat([HH, f2], dim=1)
        f2_HH = self.one_conv_f2_hh(f2_HH)

        LL_down3 = self.LL_down3(LL)
        f3_LL = torch.cat([LL_down3, f3], dim=1)
        f3_LL = self.one_conv_f3_ll(f3_LL)

        LL_down4 = self.LL_down4(LL)
        f4_LL = torch.cat([LL_down4, f4], dim=1)
        f4_LL = self.one_conv_f4_ll(f4_LL)

        prior_cam = self.GPM(x4)
        pred_0 = F.interpolate(prior_cam, size=x.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.decoder(
            [f1_HH, f2_HH, f3_LL, f4_LL], prior_cam,
            x)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1




class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class ETM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ETM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channels, out_channels, 3, 1, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0)),
        )
        self.conv_cat = BasicConv2d(4 * out_channels, out_channels, 1, padding=0)
        self.conv_res = BasicConv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x1 = F.interpolate(x1, size=(x.shape[2] // 2, x.shape[3] // 2), mode='bilinear')
        x2 = F.interpolate(x2, size=(x.shape[2] // 2, x.shape[3] // 2), mode='bilinear')
        x3 = F.interpolate(x3, size=(x.shape[2] // 2, x.shape[3] // 2), mode='bilinear')
        x4 = F.interpolate(x4, size=(x.shape[2] // 2, x.shape[3] // 2), mode='bilinear')
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class BasicDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, out_padding=0,
                 need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, dilation=dilation, output_padding=out_padding,
                                       bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x


class UNetDecoderWithEdgesBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(UNetDecoderWithEdgesBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )
        self.out_B = nn.Sequential(
            BasicDeConv2d(in_channels, mid_channels // 2, kernel_size=3, stride=2, padding=1, out_padding=1),
            BasicConv2d(mid_channels // 2, mid_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 4, 1, kernel_size=3, padding=1)
        )
        self.ode = nn.Sequential(
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1),
            BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, prior_cam):
        prior_cam = F.interpolate(prior_cam, size=x.size()[2:], mode='bilinear',
                                  align_corners=True)  # 2,1,12,12->2,1,48,48
        yt = self.conv(torch.cat([x, prior_cam.expand(-1, x.size()[1], -1, -1)], dim=1))

        ode_out = self.ode(yt)
        bound = self.out_B(ode_out)
        bound = self.edge_enhance(bound)

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        y = r_prior_cam.expand(-1, x.size()[1], -1, -1).mul(x)

        cat2 = torch.cat([y, ode_out], dim=1)  # 2,128,48,48

        y = self.out_y(cat2)
        y = y + prior_cam
        return y, bound

    def edge_enhance(self, img):
        bs, c, h, w = img.shape
        gradient = img.clone()
        gradient[:, :, :-1, :] = abs(gradient[:, :, :-1, :] - gradient[:, :, 1:, :])
        gradient[:, :, :, :-1] = abs(gradient[:, :, :, :-1] - gradient[:, :, :, 1:])
        out = img - gradient
        out = torch.clamp(out, 0, 1)
        return out


class UNetDecoderWithEdges(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(UNetDecoderWithEdges, self).__init__()

        self.REU_f1 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)
        self.REU_f2 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)
        self.REU_f3 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)
        self.REU_f4 = UNetDecoderWithEdgesBlock(in_channels, mid_channels)

    def forward(self, x, prior_0, pic):
        f1, f2, f3, f4 = x

        f4_out, bound_f4 = self.REU_f4(f4, prior_0)  # b,1,12,12 b,1,48,48
        f4 = F.interpolate(f4_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f4 = F.interpolate(bound_f4, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f3_out, bound_f3 = self.REU_f3(f3, f4_out)  # b,1,24,24 b,1,96,96
        f3 = F.interpolate(f3_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f3 = F.interpolate(bound_f3, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f2_out, bound_f2 = self.REU_f2(f2, f3_out)  # b,1,48,48 b,1,192,192
        f2 = F.interpolate(f2_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f2 = F.interpolate(bound_f2, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        f1_out, bound_f1 = self.REU_f1(f1, f2_out)  # b,1,96,96 b,1,384,384
        f1 = F.interpolate(f1_out, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384
        bound_f1 = F.interpolate(bound_f1, size=pic.size()[2:], mode='bilinear', align_corners=True)  # b,1,384,384

        return f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


class GCM3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCM3, self).__init__()
        self.T1 = ETM(in_channels[0], out_channels)
        self.T2 = ETM(in_channels[1], out_channels)
        self.T3 = ETM(in_channels[2], out_channels)
        self.T4 = ETM(in_channels[3], out_channels)

        self.decoder = nn.Conv2d(out_channels * 4, out_channels, 3, 1, 1)
        self.DWT = DWT()

    def forward(self, f1, f2, f3, f4):
        f1 = self.T1(f1)
        f2 = self.T2(f2)
        f3 = self.T3(f3)
        f4 = self.T4(f4)
        camo = self.decoder(torch.cat(
            [f1, F.upsample(f2, size=f1.shape[2:], mode='bilinear', align_corners=True),
             F.upsample(f3, size=f1.shape[2:], mode='bilinear', align_corners=True),
             F.upsample(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)], 1))
        LL, LH, HL, HH = self.DWT(camo)
        return LL, LH, HL, HH, f1, f2, f3, f4


class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)] * 4)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size = anchor.shape[-1]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size:
                x = F.adaptive_avg_pool2d(x, (target_size, target_size))
            elif x.shape[-1] < target_size:
                x = F.interpolate(x, size=(target_size, target_size),
                                  mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans


class GPM(nn.Module):
    def __init__(self, in_c=128, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=32):
        super(GPM, self).__init__()
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(in_c, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(in_c, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(in_c, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(in_c, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(in_c, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth * 5, depth, kernel_size=1, padding=0),
        )
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        size = x.shape[2:]
        branch_main = self.branch_main(x)
        branch_main = F.interpolate(branch_main, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        out = torch.cat([branch_main, branch0, branch1, branch2, branch3], 1)
        out = self.head(out)
        out = self.out(out)
        return out


if __name__ == '__main__':
    net = Network(channels=64)  # .eval()
    import thop

    x = torch.rand(1, 3, 384, 384)

    flops, params = thop.profile(net, inputs=(x,))
    from thop import clever_format

    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
