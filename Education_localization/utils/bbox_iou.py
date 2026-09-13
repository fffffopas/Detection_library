import torch

def iou(box1, box2):
    b1x1, b1y1, b1x2, b1y2 = box1
    b2x1, b2y1, b2x2, b2y2 = box2

    area1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    area2 = (b2x2 - b2x1) * (b2y2 - b2y1)

    x_left = torch.max(b2x1, b1x1)
    y_top = torch.max(b1y1, b2y1)
    x_right = torch.min(b1x2, b2x2)
    y_bottom = torch.min(b1y2, b2y2)

    if x_right < x_left or y_bottom < y_top:
        return torch.tensor(0, dtype=torch.float)

    w = x_right - x_left
    h = y_bottom - y_top

    inter = w*h
    union = area1 + area2 - inter

    iou = inter / union
    return iou

def iou_wh(boxes, clusters):
    w_min = torch.minimum(boxes[:, None, 0], clusters[None, :, 0])
    h_min = torch.minimum(boxes[:, None, 1], clusters[None, :, 1])

    intersection = w_min * h_min

    box_area = boxes[:, 0] * boxes[:, 1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    union = box_area[:, None] + cluster_area[None, :] - intersection

    return intersection/union

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, torch.newaxis, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, torch.newaxis, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, torch.newaxis] + area2 - inter

    iou = inter / union
    return iou

def bbox_iou(boxes1, boxes2):

    area1 = (boxes1[..., 2] - boxes1[..., 0]).clamp(min=0) * (boxes1[..., 3] - boxes1[..., 1]).clamp(min=0)
    area2 = (boxes2[..., 2] - boxes2[..., 0]).clamp(min=0) * (boxes2[..., 3] - boxes2[..., 1]).clamp(min=0)

    lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    wh = (rb - lt).clamp(min=0)

    inter = wh[..., 0] * wh[..., 1]

    union = area1 + area2 - inter + 1e-6

    iou = inter / union
    return iou
