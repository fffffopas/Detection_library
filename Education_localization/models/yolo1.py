import torch
import torch.nn as nn
from utils.general import clip_boxes
from utils.convert_box import xcell_ycell_wh2xyxy

class ConvBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bn=True, act="leakyrelu"):
        super().__init__()
        self.bn = bn
        self.name_act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias= not self.bn)

        if self.bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)

        self.fn_act = nn.ModuleDict({
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" : nn.LeakyReLU(inplace=True),
        })

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.batchnorm(x)

        out = self.fn_act[self.name_act](x)
        return out

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn=True, act="leakyrelu"):
        super().__init__()
        self.conv1x1 = ConvBA(in_channels, out_channels//2, (1,1), bn=bn, act=act)
        self.conv3x3 = ConvBA(out_channels//2, out_channels, (3, 3), padding=1, bn=bn, act=act)

    def forward(self, x):
        x = self.conv1x1(x)
        out = self.conv3x3(x)
        return out


class BackBone(nn.Module):
    def __init__(self, bn=True, act="leakyrelu"):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBA(3, 64, (7, 7), stride=2, padding=3, bn=bn, act=act),
            nn.MaxPool2d((2, 2), stride=2),
            ConvBA(64, 192, (3, 3), padding=1, bn=bn, act=act),
            nn.MaxPool2d((2, 2), stride=2),
            ConvBlock(192, 256, bn=bn, act=act),
            ConvBlock(256, 512, bn=bn, act=act),
            nn.MaxPool2d((2, 2), stride=2),
            ConvBlock(512, 512, bn=bn, act=act),
            ConvBlock(512, 512, bn=bn, act=act),
            ConvBlock(512, 512, bn=bn, act=act),
            ConvBlock(512, 512, bn=bn, act=act),
            ConvBlock(512, 1024, bn=bn, act=act),
            nn.MaxPool2d((2, 2), stride=2),
            ConvBlock(1024, 1024, bn=bn, act=act),
            ConvBlock(1024, 1024, bn=bn, act=act),
        )


    def forward(self, x):
        return self.layers(x)

class YOLOv1(nn.Module):

    def __init__(self, S=7, B=2, num_cl=20, backbone=None, bn=True, act="leakyrelu", use_layer=None, use_sigmoid=None, dropout=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.num_cl = num_cl
        self.name_act = act
        self.use_layer = use_layer
        self.use_sigmoid = use_sigmoid

        self.fn_act = nn.ModuleDict({
            "relu" : nn.ReLU(inplace=True),
            "leakyrelu" : nn.LeakyReLU(inplace=True)
        })

        if backbone is None:
            self.backbone = BackBone(bn=bn, act=self.name_act)
            channels=1024

        else:
            self.backbone = backbone
            for layer in backbone.modules():
                if isinstance(layer, nn.Conv2d):
                    channels = layer.out_channels

        self.conv1 = ConvBA(channels, 1024, (3, 3), padding=1, bn=bn, act=self.name_act)
        self.conv2 = ConvBA(1024, 1024, (3, 3), stride=2, padding=1, bn=bn, act=self.name_act)
        self.conv3 = ConvBA(1024, 1024, (3, 3), padding=1, bn=bn, act=self.name_act)
        self.conv4 = ConvBA(1024, 1024, (3, 3), padding=1, bn=bn, act=self.name_act)

        if self.use_layer == "conv1x1":
            self.reg_layer = nn.Conv2d(1024, 5 * self.B + self.num_cl, (1, 1))
        elif self.use_layer == "conv3x3":
            self.reg_layer = nn.Conv2d(1024, 5 * self.B + self.num_cl, (3, 3), padding=1)

        else:
            self.reg_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.S * self.S * 1024, 4096),
                self.fn_act[self.name_act],
                nn.Dropout(p=dropout),
                nn.Linear(4096, self.S * self.S * (5 * self.B + self.num_cl))

            )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.backbone(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.reg_layer(x)

        if self.use_layer:
            out = x.permute(0, 2, 3, 1)
        else:
            out = x.reshape(batch_size, self.S, self.S, -1)

        if self.use_sigmoid == "full_sigmoid":
            out = torch.nn.functional.sigmoid(out)

        elif self.use_sigmoid == "not_full_sigmoid":
            bc = torch.nn.functional.sigmoid(out[..., :5*2])
            out = torch.cat([bc, out[..., 5*2:]], dim=-1)

        return out if self.training else (out, self.bboxes_conf_clases(out))

    def bboxes_conf_clases(self, out):
        new_out = out.clone()
        batch_size = new_out.shape[0]
        boxes_conf = new_out[..., :5*self.B]
        boxes_conf = boxes_conf.reshape(batch_size, 7, 7, 2, -1)
        boxes_conf[..., 2:4] = boxes_conf[..., 2:4].square()

        boxes = xcell_ycell_wh2xyxy(boxes_conf[..., :4])
        clip_boxes(boxes)
        boxes_conf[..., :4] = boxes

        clases = new_out[..., 5*2:]
        clases = clases.reshape(batch_size, 7, 7, 1, -1)
        clases = clases.repeat(1, 1, 1, 2, 1)

        bboxes_conf_clases = torch.cat([boxes_conf, clases], dim=-1)
        bboxes_conf_clases = bboxes_conf_clases.reshape(batch_size, 7*7*2, -1)

        return bboxes_conf_clases