import torch
import torch.nn as nn
import math

class Block(nn.Module):
    def __init__(self, in_channels: int, act: str="leakyrelu", bias: bool=False):
        super().__init__()

        self.act = nn.ModuleDict(
            {
                "relu": nn.ReLU(),
                "leakyrelu" : nn.LeakyReLU(negative_slope=0.1)
            }
        )
        self.name_act = act

        self.batch_norm = nn.Sequential(
            nn.BatchNorm2d(in_channels//2),
            nn.BatchNorm2d(in_channels),
        )
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, bias=bias),
            nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=3, padding=1, bias=bias)
        )

    def forward(self, x):
        output = x
        for layer, batch_norm in zip(self.layers, self.batch_norm):
            x = self.act[self.name_act](batch_norm(layer(x)))

        return output + x

class SimpleBlock(nn.Module):
    def __init__(self, count_blocks, in_channels: int, act: str="leakyrelu", bias: bool=False):
        super().__init__()

        act_func = {
                "relu" : nn.ReLU(),
                "leakyrelu" : nn.LeakyReLU(negative_slope=0.1),
            }

        variants = [[1, in_channels//2, 0], [3, in_channels, 1]]
        len_variants = len(variants)
        list_layers = []
        v = 0
        last_chan = in_channels

        for i in range(count_blocks):
            temp = v % len_variants
            kernel_size, chan, padding = variants[temp]
            list_layers.append(nn.Conv2d(last_chan, chan, kernel_size=kernel_size, padding=padding, bias=bias))
            list_layers.append(nn.BatchNorm2d(chan))
            list_layers.append(act_func[act])

            last_chan = chan
            v += 1

        self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.layers(x)
                
class FullBlock(nn.Module):
    def __init__(self, count_blocks: int, in_channels: int, act: str="leakyrelu", bias: bool=False, down_sample_conv: bool=True):
        super().__init__()


        list_layers = []
        for i in range(count_blocks):
            list_layers.append(Block(in_channels, act, bias))

        if down_sample_conv:

            act_func = nn.ModuleDict(
                {
                    "relu" : nn.ReLU(),
                    "leakyrelu" : nn.LeakyReLU(negative_slope=0.1),
                }
            )

            list_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=3, padding=1, stride=2))
            list_layers.append(nn.BatchNorm2d(in_channels*2))
            list_layers.append(act_func[act])

        self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.layers(x)

class Darknet_53S(nn.Module):
    def __init__(self, in_channels: int=3, act: str="leakyrelu", bias: bool=False):
        super().__init__()
        # 416x416
        act_func = nn.ModuleDict(
            {
                "relu" : nn.ReLU(),
                "leakyrelu" : nn.LeakyReLU(negative_slope=0.1),
            }
        )

        self.layers_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, bias=bias), # 416x416
            nn.BatchNorm2d(32),
            act_func[act],
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2, bias=bias), # 208x208
            nn.BatchNorm2d(64),
            act_func[act],
            FullBlock(count_blocks=1, in_channels=64, act=act, bias=bias), # 104x104
            FullBlock(count_blocks=2, in_channels=128, act=act, bias=bias) # 52x52 -> Забираем отсюда
        )

        self.layers_2 = FullBlock(count_blocks=8, in_channels=256, act=act, bias=bias) # 26x26 -> Забираем отсюда
        self.layers_3 = nn.Sequential(
            FullBlock(count_blocks=8, in_channels=512, act=act, bias=bias), # 13x13
            FullBlock(count_blocks=4, in_channels=1024, act=act, down_sample_conv=False, bias=bias), # 13x13 забираем отсюда
        )

    def forward(self, x):
        out1 = self.layers_1(x)
        out2 = self.layers_2(out1)
        out3 = self.layers_3(out2)

        return out1, out2, out3

class Darknet_53(nn.Module):
    def __init__(self, in_channels: int=3, act: str="leakyrelu", bias: bool=False, classes: int=10):
        super().__init__()

        self.classes = classes
        self.darknet_s = Darknet_53S(in_channels=in_channels, act=act, bias=bias)
        self.connection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, self.classes)
        )
        
    def forward(self, x):
        return self.connection(self.darknet_s(x)[2])

class YOLOv3(nn.Module):
    def __init__(self, in_channels: int=3, act: str="leakyrelu", bias: bool=False, num_classes: int=80, darknet_input: nn.Module=None):
        super().__init__()

        act_func = nn.ModuleDict(
            {
                "relu" : nn.ReLU(),
                "leakyrelu" : nn.LeakyReLU(negative_slope=0.1),
            }
        )
        self.num_classes = num_classes
        if darknet_input is not None:
            self.darknet = darknet_input
        else:
            self.darknet = Darknet_53S(in_channels, act, bias) # 256x52x52, 512x26x26, 1024x13x13
        self.upsample_13_26 = nn.Upsample(scale_factor=2, mode="nearest") 

        self.det_con_13 = SimpleBlock(count_blocks=5, in_channels=1024, bias=bias, act=act) # 512x13x13
        self.detection_head_13 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(1024),
            act_func[act],
            nn.Conv2d(in_channels=1024, out_channels=(3*(5 + self.num_classes)), kernel_size=1, bias=True) # -> выход 1
        )

        self.to_26 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            act_func[act],
            self.upsample_13_26, #256x26x26
        )


        self.det_con_26 = SimpleBlock(count_blocks=5, in_channels=768, bias=bias, act=act) # 384x26x26
        self.detection_head_26 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=768, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(768),
            act_func[act],
            nn.Conv2d(in_channels=768, out_channels=(3 * (5 + self.num_classes)), kernel_size=1, bias=True) # -> выход 2
        )

        self.to_52 = nn.Sequential(
                    nn.Conv2d(in_channels=384, out_channels=192, kernel_size=1),
                    nn.BatchNorm2d(192),
                    act_func[act],
                    self.upsample_13_26, # 192x52x52
        )


        self.detection_head_52 = nn.Sequential(
            SimpleBlock(count_blocks=5, in_channels=448, bias=bias), # 224x52x52
            nn.Conv2d(in_channels=224, out_channels=448, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(448),
            act_func[act],
            nn.Conv2d(in_channels=448, out_channels=(3 * (5 + self.num_classes)), kernel_size=1, bias=True) # -> выход 3
        )

        for head in (self.detection_head_13, self.detection_head_26, self.detection_head_52):
            b = head[-1].bias.view(3, -1)
            b.data[:, 4] += math.log(0.01 / 0.99)
            b.data[:, 5:] += math.log(0.01 / 0.99)
            head[-1].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        grid_52, grid_26, grid_13 = self.darknet(x)

        temp_13 = self.det_con_13(grid_13)
        large_objects = self.detection_head_13(temp_13) # -> 1

        grid_26 = torch.concatenate([grid_26, self.to_26(temp_13)], dim=1)
        temp_26 = self.det_con_26(grid_26)
        medium_objects = self.detection_head_26(temp_26) # -> 2

        grid_52 = torch.concatenate([grid_52, self.to_52(temp_26)], dim=1)
        small_objects = self.detection_head_52(grid_52) # -> 3

        return small_objects, medium_objects, large_objects


