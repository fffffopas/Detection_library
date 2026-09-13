import torch
from .general import grid_xy

def xywh2xyxy(inp):

    out = torch.empty_like(inp)

    xy = inp[..., :2]
    wh = inp[..., 2:]/2

    out[..., :2] = xy - wh
    out[..., 2:] = xy + wh

    return out

def xyxy2xywh(inp):

    out = torch.empty_like(inp)

    out[..., 0] = (inp[..., 0] + inp[..., 2])/ 2
    out[..., 1] = (inp[..., 1] + inp[..., 3])/ 2

    out[..., 2] = inp[..., 2] - inp[..., 0]
    out[..., 3] = inp[..., 3] - inp[..., 1]

    return out

def xcell_ycell_wh2xyxy(inp, size, count, anchors):
   
    out = torch.empty_like(inp, dtype=inp.dtype)

    grid = grid_xy(dtype=inp.dtype, size=size, count_box_per_pred=count, device=inp.device)

    xy = inp[..., :2]/size + grid  

    anchors_r = anchors.reshape(1, 1, 1, 3, 2).to(inp.dtype)
    wh_full = anchors_r * torch.exp(inp[..., 2:4]).clamp(max=10)   
    wh = wh_full / 2

    out[..., :2] = xy - wh
    out[..., 2:4] = xy + wh

    return out