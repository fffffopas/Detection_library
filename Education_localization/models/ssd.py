import torch
import torch.nn as nn

class BlockConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, count_blocks: int=3, name_act: str="relu", need_pool: bool=True, ceil_mode: bool=True):
        assert (count_blocks > 0), "Количество блоков должно быть целым числом, большим нуля"
        super().__init__()

        self.act_names = nn.ModuleDict({
            "relu": nn.ReLU(),
            "leakyrelu" : nn.LeakyReLU(),
        })
        list_layers = []

        list_layers.append(nn.Conv2d(kernel_size=3,
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    padding=1,
                                    ))
        list_layers.append(self.act_names[name_act])


        for i in range(count_blocks-1):
            list_layers.append(nn.Conv2d(kernel_size=3,
                                        in_channels=out_channels,
                                        out_channels=out_channels,
                                        padding=1,
                                        ))
            list_layers.append(self.act_names[name_act])

        if need_pool:
            list_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode))

        self.layers = nn.Sequential(*list_layers)

    def forward(self, x):
        return self.layers(x)

class VGG16S(nn.Module):
    def __init__(self, in_channels: int=3, name_act: str="relu", ceil_mode: bool=True):
        super().__init__()
        self.layers = nn.Sequential(
            BlockConv(in_channels=in_channels, out_channels=64, count_blocks=2, name_act=name_act, ceil_mode=ceil_mode),
            BlockConv(in_channels=64, out_channels=128, count_blocks=2, name_act=name_act, ceil_mode=ceil_mode),
            BlockConv(in_channels=128, out_channels=256, name_act=name_act, ceil_mode=ceil_mode),
            BlockConv(in_channels=256, out_channels=512, name_act=name_act, need_pool=False, ceil_mode=ceil_mode),
        )

    def forward(self, x):
        return self.layers(x)

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=20):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = x/norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * x
        return out


class SSD(nn.Module):
    def __init__(self, in_channels: int=3, name_act: str="relu", pred_trained_block: nn.Module=None, num_classes: int=20):
        super().__init__()
        self.act_names = nn.ModuleDict({
            "relu": nn.ReLU(),
            "leakyrelu" : nn.LeakyReLU(),
        })
        self.anchors = [4, 6, 6 ,6, 4, 4]
        self.num_classes = num_classes + 1
        self.l2 = L2Norm(512)

        if pred_trained_block:
            self.vgg16 = pred_trained_block
        else:
            self.vgg16 = VGG16S(in_channels, name_act=name_act)

        block_1 = nn.Sequential(
            self.vgg16,
        )
        loc_1 = nn.Conv2d(512, self.anchors[0] * 4, kernel_size=3, padding=1)
        conf_1 = nn.Conv2d(512, self.anchors[0] * self.num_classes, kernel_size=3, padding=1)

        block_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            BlockConv(in_channels=512, out_channels=512, name_act=name_act, need_pool=False),
            self.act_names[name_act],
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(kernel_size=3, in_channels=512, out_channels=1024, dilation=6, padding=6),
            self.act_names[name_act],
            nn.Conv2d(kernel_size=1, in_channels=1024, out_channels=1024),
            self.act_names[name_act],
        )
        loc_2 = nn.Conv2d(1024, self.anchors[1] * 4, kernel_size=3, padding=1)
        conf_2 = nn.Conv2d(1024, self.anchors[1] * self.num_classes, kernel_size=3, padding=1)

        block_3 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=1024, out_channels=256),
            self.act_names[name_act],
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=512, stride=2, padding=1),
            self.act_names[name_act],
        )
        loc_3 = nn.Conv2d(512, self.anchors[2] * 4, kernel_size=3, padding=1)
        conf_3 = nn.Conv2d(512, self.anchors[2] * self.num_classes, kernel_size=3, padding=1)

        block_4 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=512, out_channels=128),
            self.act_names[name_act],
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=2, padding=1),
            self.act_names[name_act],
        )
        loc_4 = nn.Conv2d(256, self.anchors[3] * 4, kernel_size=3, padding=1)
        conf_4 = nn.Conv2d(256, self.anchors[3] * self.num_classes, kernel_size=3, padding=1)

        block_5 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128),
            self.act_names[name_act],
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=0),
            self.act_names[name_act]
        )
        loc_5 = nn.Conv2d(256, self.anchors[4] * 4, kernel_size=3, padding=1)
        conf_5 = nn.Conv2d(256, self.anchors[4] * self.num_classes, kernel_size=3, padding=1)

        block_6 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=256, out_channels=128),
            self.act_names[name_act],
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256, stride=1, padding=0)
        )

        loc_6 = nn.Conv2d(256, self.anchors[5] * 4, kernel_size=3, padding=1)
        conf_6 = nn.Conv2d(256, self.anchors[5] * self.num_classes, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([block_1, block_2, block_3, block_4, block_5, block_6])
        self.loc = nn.ModuleList([loc_1, loc_2, loc_3, loc_4, loc_5, loc_6])
        self.conf = nn.ModuleList([conf_1, conf_2, conf_3, conf_4, conf_5, conf_6])

    def _create_output(self, loc, conf, output):
        loc_out, conf_out = loc(output), conf(output)
        batch_size= loc_out.shape[0]

        loc_out = loc_out.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        conf_out = conf_out.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
        return loc_out, conf_out

    def forward(self, x):
        x = self.layers[0](x)
        x_step = self.l2(x)
        loc, conf = self._create_output(self.loc[0], self.conf[0], x_step)

        loc_s = [loc]
        conf_s = [conf]
        for i in range(1, len(self.layers)):
            x = self.layers[i](x)
            loc, conf = self._create_output(self.loc[i], self.conf[i], x)
            loc_s.append(loc)
            conf_s.append(conf)

        clear_loc = torch.cat(loc_s, dim=1)
        clear_conf = torch.cat(conf_s, dim=1)
        return clear_loc, clear_conf
        



        

        
