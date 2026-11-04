# Source: https://github.com/Megvii-BaseDetection/YOLOX (Apache-2.0), yolox/models: network_blocks.py, darknet.py,
# yolo_pafpn.py, yolo_head.py. Inference-only copy: training losses, the Darknet53/ResNet backbones and
# configurable activations are removed. Module and parameter names are kept so official checkpoints load as is.

import torch
import torch.nn as nn


class BaseConv(nn.Module):
    """Conv2d -> BatchNorm -> SiLU."""

    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise conv followed by pointwise conv."""

    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize, stride, groups=in_channels)
        self.pconv = BaseConv(in_channels, out_channels, 1, 1)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = conv(hidden_channels, out_channels, 3, 1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.use_add else y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP."""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        self.conv2 = BaseConv(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class CSPLayer(nn.Module):
    """C3 in YOLOv5: CSP bottleneck with 3 convolutions."""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise) for _ in range(n)]
        )

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride)

    def forward(self, x):
        top_left = x[..., ::2, ::2]
        top_right = x[..., ::2, 1::2]
        bot_left = x[..., 1::2, ::2]
        bot_right = x[..., 1::2, 1::2]
        return self.conv(torch.cat((top_left, bot_left, top_right, bot_right), dim=1))


class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False):
        super().__init__()
        self.out_features = out_features
        conv = DWConv if depthwise else BaseConv
        base_channels = int(wid_mul * 64)
        base_depth = max(round(dep_mul * 3), 1)

        self.stem = Focus(3, base_channels, ksize=3)
        self.dark2 = nn.Sequential(
            conv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise),
        )
        self.dark3 = nn.Sequential(
            conv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise),
        )
        self.dark4 = nn.Sequential(
            conv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise),
        )
        self.dark5 = nn.Sequential(
            conv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        for name in ("dark2", "dark3", "dark4", "dark5"):
            x = getattr(self, name)(x)
            outputs[name] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class YOLOPAFPN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_channels=(256, 512, 1024), depthwise=False):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise)
        self.in_features = ("dark3", "dark4", "dark5")
        conv = DWConv if depthwise else BaseConv
        c0, c1, c2 = (int(c * width) for c in in_channels)
        n = round(3 * depth)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(c2, c1, 1, 1)
        self.C3_p4 = CSPLayer(2 * c1, c1, n, False, depthwise=depthwise)
        self.reduce_conv1 = BaseConv(c1, c0, 1, 1)
        self.C3_p3 = CSPLayer(2 * c0, c0, n, False, depthwise=depthwise)
        self.bu_conv2 = conv(c0, c0, 3, 2)
        self.C3_n3 = CSPLayer(2 * c0, c1, n, False, depthwise=depthwise)
        self.bu_conv1 = conv(c1, c1, 3, 2)
        self.C3_n4 = CSPLayer(2 * c1, c2, n, False, depthwise=depthwise)

    def forward(self, x):
        features = self.backbone(x)
        x2, x1, x0 = (features[f] for f in self.in_features)

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.C3_p4(torch.cat([self.upsample(fpn_out0), x1], 1))
        fpn_out1 = self.reduce_conv1(f_out0)
        pan_out2 = self.C3_p3(torch.cat([self.upsample(fpn_out1), x2], 1))
        pan_out1 = self.C3_n3(torch.cat([self.bu_conv2(pan_out2), fpn_out1], 1))
        pan_out0 = self.C3_n4(torch.cat([self.bu_conv1(pan_out1), fpn_out0], 1))
        return pan_out2, pan_out1, pan_out0


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=(8, 16, 32), in_channels=(256, 512, 1024), depthwise=False):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        conv = DWConv if depthwise else BaseConv
        hidden = int(256 * width)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for channels in in_channels:
            self.stems.append(BaseConv(int(channels * width), hidden, 1, 1))
            self.cls_convs.append(nn.Sequential(conv(hidden, hidden, 3, 1), conv(hidden, hidden, 3, 1)))
            self.reg_convs.append(nn.Sequential(conv(hidden, hidden, 3, 1), conv(hidden, hidden, 3, 1)))
            self.cls_preds.append(nn.Conv2d(hidden, num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(hidden, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(hidden, 1, 1, 1, 0))

    def forward(self, features):
        """Return (batch, anchors, 5 + num_classes): cx, cy, w, h in input pixels, objectness, class scores."""
        outputs, grids, strides = [], [], []
        for k, (x, stride) in enumerate(zip(features, self.strides, strict=True)):
            x = self.stems[k](x)
            cls_output = self.cls_preds[k](self.cls_convs[k](x)).sigmoid()
            reg_feat = self.reg_convs[k](x)
            output = torch.cat([self.reg_preds[k](reg_feat), self.obj_preds[k](reg_feat).sigmoid(), cls_output], 1)
            outputs.append(output.flatten(start_dim=2))

            height, width = x.shape[-2:]
            yv, xv = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            grids.append(torch.stack((xv, yv), 2).view(-1, 2))
            strides.append(torch.full((height * width, 1), stride))

        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        grid = torch.cat(grids).to(outputs)
        stride = torch.cat(strides).to(outputs)
        xy = (outputs[..., :2] + grid) * stride
        wh = torch.exp(outputs[..., 2:4]) * stride
        return torch.cat([xy, wh, outputs[..., 4:]], dim=-1)


class YOLOX(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def yolox_nano(num_classes: int = 80) -> YOLOX:
    """YOLOX-Nano as in exps/default/yolox_nano.py: depth 0.33, width 0.25, depthwise convolutions."""
    depth, width = 0.33, 0.25
    model = YOLOX(YOLOPAFPN(depth, width, depthwise=True), YOLOXHead(num_classes, width, depthwise=True))
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3  # trained with eps 1e-3 (init_yolo); the default 1e-5 changes outputs
            module.momentum = 0.03
    return model
