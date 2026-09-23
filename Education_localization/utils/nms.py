import torch
from .bbox_iou import box_iou
from .convert_box import xywh2xyxy
from torch.nn import functional as F

def nms(boxes, scores, threshold=0.5):
    _, sorted_idx = scores.sort(descending=True)

    keep = []
    while sorted_idx.numel() > 0:
        if sorted_idx.numel() == 1:
            keep.append(sorted_idx)
            break

        idx = sorted_idx[0]
        keep.append(idx)

        boxs1 = boxes[sorted_idx[0]].unsqueeze(dim=0)
        boxs2 = boxes[sorted_idx[1:]]
        iou = box_iou(boxs1, boxs2)

        i = (iou < threshold).nonzero()[:, 1]
        if i.numel() == 0:
            break

        sorted_idx = sorted_idx[i+1]

    return torch.tensor(keep, dtype=torch.int)

def non_max_supression(pred, score_threshold=0.25, iou_threshold=0.45, agnostic=False, max_wh=7600, classes=None):

    if classes is not None:
        classes = torch.tensor(classes, device=pred.device)

    bs = pred.shape[0]
    nc = pred.shape[1] - 4
    candidates = pred[:, 4] > score_threshold
    pred = pred.transpose(-1, -2)
    pred = torch.cat((xywh2xyxy(pred[..., :4]), pred[..., 4:]), dim=-1)

    output = [torch.zeros((0, 6), device=pred.device)] * bs

    for idx, pr in enumerate(pred):

        pr = pr[candidates[idx]]

        if not pr.shape[0]:
            continue

        box, cls = pr.split((4, nc), dim=1)
        score, idx_cls = cls.max(dim=1, keepdim=True)

        pr = torch.cat((box, score, idx_cls.float()), dim=1)

        if classes is not None:
            i = (pr[:, 5:6] == classes).any(dim=1)
            pr = pr[i]

        if not pr.shape[0]:
            continue

        pr[:, 5:] *= pr[:, 4:5]

        scores = pr[:, 4]
        scaling_for_classes = pr[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pr[:, :4] + scaling_for_classes

        idx_nms = nms(boxes, scores, iou_threshold)
        output[idx] = pr[idx_nms]

    return output

def nms_yolov1(
        pred,
        score_threshold=0.25,
        iou_threshold=0.45,
        agnostic=False,
        max_wh=7600,
        classes=None

):

    if classes is not None:
        classes = torch.tensor(classes, device=pred.device)

    bs = pred.shape[0]
    # nc = pred.shape[1] - 5
    candidates = pred[..., 4] > score_threshold  # (bs, nb)

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    
    for idx, pr in enumerate(pred):
       
        pr = pr[candidates[idx]]

        
        if not pr.shape[0]:
            continue
        
        pr[:, 5:] *= pr[:, 4:5]
        
        boxes = pr[..., :4]
        cls = pr[:, 5:]

        
        score, idx_cls = cls.max(dim=1, keepdim=True)

        
        pr = torch.cat((boxes, score, idx_cls.float()), dim=1)[score.view(-1) > score_threshold]

     
        if classes is not None:
            i = (pr[:, 5:6] == classes).any(dim=1)
            pr = pr[i]

        
        if not pr.shape[0]:
            continue

        scores = pr[:, 4]
        scaling_for_classes = pr[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pr[:, :4] + scaling_for_classes

        idx_nms = nms(boxes, scores, iou_threshold)

        output[idx] = pr[idx_nms]

    return output


def nms_ssd(boxes_all, conf, score_threshold=0.5, iou_threshold=0.45):
    bs = boxes_all.shape[0]
    boxes_all = xywh2xyxy(boxes_all)
    output = [torch.zeros((0, 6), device=boxes_all.device)] * bs

    probs_all = F.softmax(conf, dim=-1)

    for idx in range(bs):
        boxes = boxes_all[idx]
        probs = probs_all[idx]

        scores, labels = probs[:, 1:].max(dim=1)
        mask = scores > score_threshold

        boxes_i, scores_i, labels_i = boxes[mask], scores[mask], labels[mask]

        if boxes_i.shape[0] == 0:
            continue

        keep = nms(boxes_i, scores_i, iou_threshold)

        result = torch.cat([boxes_i[keep],
                            scores_i[keep].unsqueeze(1),
                            labels_i[keep].unsqueeze(1).float()
                        ], dim=1)

        output[idx] = result

    return output

def nms_yolov3(pred, score_threshold=0.25, iou_threshold=0.45, agnostic=False, max_wh=7600, classes=None):
    if classes is not None:
        classes = torch.tensor(classes, device=pred.device)

    bs = pred.shape[0]

    pred = torch.cat((xywh2xyxy(pred[..., :4]), pred[..., 4:]), dim=-1)
    candidates = pred[..., 4] > score_threshold
    output = [torch.zeros((0, 6), device=pred.device)] * bs

    for idx, pr in enumerate(pred):
        pr = pr[candidates[idx]]

        if not pr.shape[0]:
            continue

        pr[:, 5:] *= pr[:, 4:5]
        boxes = pr[..., :4]
        cls = pr[:, 5:]

        score, idx_cls = cls.max(dim=1, keepdim=True)

        pr = torch.cat((boxes, score, idx_cls.float()), dim=1)[score.view(-1) > score_threshold]

        if classes is not None:
            i = (pr[:, 5:6] == classes).any(dim=1)
            pr = pr[i]

        if not pr.shape[0]:
            continue

        scores = pr[:, 4]
        scaling_for_classes = pr[:, 5:6] * (0 if agnostic else max_wh)
        boxes = pr[:, :4] + scaling_for_classes
        idx_nms = nms(boxes, scores, iou_threshold)

        output[idx] = pr[idx_nms]

    return output

def nms_vectorized(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    boxes_sorted = boxes[order]

    iou_matrix = box_iou(boxes_sorted, boxes_sorted)
    iou_matrix = iou_matrix.triu(diagonal=1)
    suppressed = torch.zeros(len(boxes_sorted), dtype=torch.bool, device=boxes.device)

    for i in range(len(boxes_sorted)):
        if suppressed[i]:
            continue

        supress_mask = iou_matrix[i] > iou_threshold
        suppressed |= supress_mask

    keep_sorted = (~suppressed).nonzero(as_tuple=True)[0]
    return order[keep_sorted]

def nms_yolov3_vectorized(pred, score_threshold=0.25, iou_threshold=0.45, agnostic=False, max_wh=7600, classes=None, max_nms=1000, max_det=300):
    device = pred.device

    if classes is not None:
        classes = torch.tensor(classes, device=device)

    B, N, _ = pred.shape
    pred = torch.cat((xywh2xyxy(pred[..., :4]), pred[..., 4:]), dim=-1)

    img_idx = torch.arange(B, device=device).view(B, 1).expand(B, N).reshape(-1)
    pred_flat = pred.reshape(-1, pred.shape[-1])

    mask = pred_flat[:, 4] > score_threshold
    pred_flat = pred_flat[mask]
    img_idx = img_idx[mask]

    if pred_flat.shape[0] > max_nms:
        top_idx = pred_flat[:, 4].argsort(descending=True)[:max_nms]
        pred_flat = pred_flat[top_idx]
        img_idx = img_idx[top_idx]

    if pred_flat.shape[0] == 0:
        return [torch.zeros((0, 6)) for _ in range(B)]

    pred_flat[:, 5:] *= pred_flat[:, 4:5]
    boxes = pred_flat[:, :4]
    cls_scores = pred_flat[:, 5:]

    scores, idx_cls = cls_scores.max(dim=1)

    keep_score = scores > score_threshold
    boxes = boxes[keep_score]
    scores = scores[keep_score]
    idx_cls = idx_cls[keep_score]
    img_idx = img_idx[keep_score]

    if classes is not None:
        keep_cls = (idx_cls.unsqueeze(1) == classes).any(dim=1)
        boxes, scores, idx_cls, img_idx = boxes[keep_cls], scores[keep_cls], idx_cls[keep_cls], img_idx[keep_cls]

    if boxes.shape[0] == 0:
        return [torch.zeros((0, 6), device=device) for _ in range(B)]

    group_id = img_idx * 20 + idx_cls.float()
    scaling = group_id.unsqueeze(1) * (0 if agnostic else max_wh)
    boxes_shifted = boxes + scaling

    keep = nms_vectorized(boxes_shifted, scores, iou_threshold)
    final = torch.cat([boxes[keep], scores[keep].unsqueeze(1), idx_cls[keep].float().unsqueeze(1)], dim=1)
    final_img_idx = img_idx[keep]

    if final.shape[0] > max_det:
        top_idx = final[:, 4].argsort(descending=True)[:max_det]
        final = final[top_idx]
        final_img_idx = final_img_idx[top_idx]

    output = [torch.zeros((0,6), device=device) for _ in range(B)]
    for b in range(B):
        output[b] = final[final_img_idx == b]

    return output

def nms_inter_fast(boxes, scores, iou_threshold):
    order = scores.argsort(descending=True)
    boxes_sorted = boxes[order]

    iou_matrix = box_iou(boxes_sorted, boxes_sorted).triu(diagonal=1)
    suppressed = (iou_matrix > iou_threshold).any(dim=0)

    keep_sorted = (~suppressed).nonzero(as_tuple=True)[0]
    return order[keep_sorted]
        