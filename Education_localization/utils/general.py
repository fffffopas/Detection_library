import torch
import numpy as np
from .bbox_iou import box_iou


def clip_boxes(boxes, shape=[1,1]):
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])
        boxes[..., 1].clamp_(0, shape[0])
        boxes[..., 2].clamp_(0, shape[1])
        boxes[..., 3].clamp_(0, shape[0])

    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def grid_xy(dtype=torch.int32, device='cpu', size: int = 52, count_box_per_pred: int=3):
    shifts_x = torch.arange(0, size, dtype=dtype, device=device) / size
    shifts_y = torch.arange(0, size, dtype=dtype, device=device) / size   

    shifts_x, shifts_y = torch.meshgrid(shifts_x, shifts_y, indexing="xy")

    shifts_x = shifts_x.reshape((1, size, size, 1, 1))   
    shifts_y = shifts_y.reshape((1, size, size, 1, 1))   

    shifts_x = shifts_x.repeat(1, 1, 1, count_box_per_pred, 1)  
    shifts_y = shifts_y.repeat(1, 1, 1, count_box_per_pred, 1)   

    grid = torch.cat([shifts_x, shifts_y], dim=-1)  
    return grid.to(device)


def process_batch(detections, labels, iouv):
    
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
