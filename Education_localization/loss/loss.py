import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.convert_box import xcell_ycell_wh2xyxy
from utils.bbox_iou import bbox_iou, box_iou

class YOLOv1Loss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def _transform_target(self, batch_size, target):
        new_target = torch.zeros(batch_size, self.S, self.S, 5 * self.B + self.C)
        target_copy = [target[i].clone() for i in range(len(target))]

        n_l = [t.shape[0] for t in target_copy]
        cls_labels = [torch.zeros(l, self.C) for l in n_l]
        for i in range(len(cls_labels)):
            cls_labels[i][torch.arange(n_l[i]), target_copy[i][:, -1].long()] = 1

        conf = [torch.ones(l, 1) for l in n_l]

        w_cell = 1 / self.S
        h_cell = 1 / self.S

        wh_cell = torch.tensor([w_cell, h_cell])
        idxs = [torch.floor(target_copy[i][:, :2] / wh_cell) for i in range(len(target_copy))]
        ijs = [torch.tensor_split(idxs[i].long(), 2, dim=1) for i in range(len(idxs))]

        for i in range(len(target_copy)):
            target_copy[i][:, :2] = (target_copy[i][:, :2] % wh_cell) / wh_cell
            bboxes_conf_labels = torch.cat([target_copy[i][:, :-1], conf[i], target_copy[i][:, :-1], conf[i], cls_labels[i]], dim=1)
            new_target[i, ijs[i][1], ijs[i][0]] = bboxes_conf_labels[:, torch.newaxis, :]

        return new_target

    def forward(self, pred, target):
        device = pred.device
        batch_size = pred.shape[0]
        target = self._transform_target(batch_size, target)

        pred_boxes_conf = pred[..., :5*self.B]
        pred_boxes_conf = pred_boxes_conf.reshape(batch_size, self.S, self.S, self.B, -1)
        pred_boxes = pred_boxes_conf[..., :4]
        xy = pred_boxes[..., :2]
        wh = pred_boxes[..., 2:].square()

        boxes_xywh = torch.cat([xy, wh], dim=-1)
        pred_class = pred[..., 5 * self.B:]

        mask_obj = torch.any((target > 0), dim=-1)

        target_boxes_conf = target[..., :5 * self.B]
        target_boxes_conf = target_boxes_conf.reshape(batch_size, self.S, self.S, self.B, -1)
        target_boxes = target_boxes_conf[..., :4]
        target_class = target[..., 5 * self.B:]

        pred_boxes_xyxy = xcell_ycell_wh2xyxy(boxes_xywh)
        target_boxes_xyxy = xcell_ycell_wh2xyxy(target_boxes)

        iou = bbox_iou(pred_boxes_xyxy, target_boxes_xyxy)
        max_idx = iou.argmax(dim=-1, keepdims=True)

        box_idx = torch.zeros(iou.shape, device=device).scatter(-1, max_idx, 1.0)
        obj_box_idx = (mask_obj[..., torch.newaxis] * box_idx).bool()
        no_obj_box_idx = ~obj_box_idx

        obj_box_pred_boxes = pred_boxes[obj_box_idx]
        obj_box_target_boxes = target_boxes[obj_box_idx]

        pred_x = obj_box_pred_boxes[..., 0]
        target_x = obj_box_target_boxes[..., 0]

        pred_y = obj_box_pred_boxes[..., 1]
        target_y = obj_box_target_boxes[..., 1]

        pred_w_sqrt = obj_box_pred_boxes[..., 2]
        target_w_sqrt = obj_box_target_boxes[..., 2].sqrt()

        pred_h_sqrt = obj_box_pred_boxes[..., 3]
        target_h_sqrt = obj_box_target_boxes[..., 3].sqrt()

        x_loss = (pred_x - target_x).square().sum()
        y_loss = (pred_y - target_y).square().sum()
        w_sqrt_loss = (pred_w_sqrt - target_w_sqrt).square().sum()
        h_sqrt_loss = (pred_h_sqrt - target_h_sqrt).square().sum()

        xywh_loss = x_loss + y_loss + w_sqrt_loss + h_sqrt_loss

        pred_conf = pred_boxes_conf[..., 4][obj_box_idx]
        target_conf = iou[obj_box_idx]

        conf_loss = (pred_conf - target_conf).square().sum()

        pred_class = pred_class[mask_obj]
        target_class = target_class[mask_obj]

        class_loss = (pred_class - target_class).square().sum()

        pred_no_obj_conf = pred_boxes_conf[..., 4][no_obj_box_idx]
        no_obj_conf_loss = pred_no_obj_conf.square().sum()

        loss = (self.lambda_coord * xywh_loss 
                + conf_loss
                + class_loss
                + self.lambda_noobj * no_obj_conf_loss)

        loss = loss / batch_size

        loss_item = loss.item()
        x_loss = (x_loss/batch_size).item()
        y_loss = (y_loss/batch_size).item()
        w_loss = (w_sqrt_loss/batch_size).item()
        h_loss = (h_sqrt_loss/batch_size).item()
        conf_loss = (conf_loss/batch_size).item()
        class_loss = (class_loss/batch_size).item()
        no_obj_conf_loss = (no_obj_conf_loss/batch_size).item()

        return loss, loss_item, x_loss, y_loss, w_loss, h_loss, conf_loss, class_loss, no_obj_conf_loss



class SSDLoss(nn.Module):
    def __init__(self, anchors: torch.Tensor, alpha: int=1):
        super().__init__()
        self.alpha = alpha
        self.anchors = anchors
        
    def _smooth(self, target, pred):
        diff = torch.abs(target - pred)

        mask_1 = diff < 1
        mask_2 = diff >= 1

        diff[mask_1] = 0.5 * diff[mask_1].square()
        diff[mask_2] -= 0.5

        return diff.sum()

    def _encode(self, matched_boxes): 
        g_encoded = torch.zeros_like(matched_boxes)
        g_encoded[:, :2] = (matched_boxes[:, :2] - self.anchors[:, :2])/ self.anchors[:, 2:]
        g_encoded[:, 2:] = torch.log(matched_boxes[:, 2:]/ self.anchors[:, 2:])
        return g_encoded

    def _decode(self, loc):
        final_boxes = torch.zeros_like(loc)
        anchors = self.anchors.unsqueeze(dim=0)
        final_boxes[..., :2] = loc[..., :2] * anchors[..., 2:] + anchors[..., :2]
        final_boxes[..., 2:] = torch.exp(loc[..., 2:]) * anchors[..., 2:]

        return final_boxes

    def forward(self, loc, conf, target):
        total_num = 0
        loss = 0
        for i, t in enumerate(target):
            boxes = t[:, :4]
            labels = t[:, 4].long()

            iou_matrix = box_iou(boxes, self.anchors)
            best = iou_matrix.argmax(dim=1)

            best_gt_iou, best_gt_idx = iou_matrix.max(dim=0)
            mask = best_gt_iou > 0.5
            mask[best] = True
            best_gt_idx[best] = torch.arange(len(t))

            matched_boxes = boxes[best_gt_idx]
            matched_labels = labels[best_gt_idx]

            g_encoded = self._encode(matched_boxes)
            loc_loss = self._smooth(loc[i][mask], g_encoded[mask]) #1

            conf_target = torch.zeros(len(self.anchors), dtype=torch.long)
            conf_target[mask] = matched_labels[mask] + 1
            conf_loss_all = F.cross_entropy(conf[i], conf_target, reduction="none")
            pos_conf_loss = conf_loss_all[mask].sum() #2

            neg_loss = conf_loss_all.clone()
            neg_loss[mask] = 0
            num_pos = mask.sum()
            num_neg = torch.clamp(3 * num_pos, max=len(self.anchors) - num_pos)
            neg_loss_sorted, _ = neg_loss.sort(descending=True)
            hard_neg_loss = neg_loss_sorted[:num_neg].sum() #3

            loss += (loc_loss + pos_conf_loss + self.alpha * hard_neg_loss)
            total_num += num_pos

        loss = loss/ total_num.clamp(min=1)
        return loss


class YOLOv3Loss(nn.Module):
    def __init__(self, list_size: list=[52, 26, 13], C: int=20, anchors: torch.Tensor=None):
        super().__init__()
        self.list_size = list_size
        self.C = C
        self.lambda_coord = 1
        self.lambda_noobj = 1
        self.register_buffer("anchors", anchors)
        self.grid = self._create_anchor_grid()
        

    def _create_anchor_grid(self):
        for i in range(len(self.list_size)):
            size = self.list_size[i]
            device = self.anchors.device

            anchors_i = self.anchors[3*i: 3*i + 3].reshape(1, 1, 1, 3, 2)
            anchors_i = anchors_i.repeat(1, size, size, 1, 1)
            x = torch.arange(0, size, dtype=torch.float32, device=device)
            y = torch.arange(0, size, dtype=torch.float32, device=device)
            x, y = torch.meshgrid([x, y], indexing="xy")

            x = x.reshape(1, size, size, 1, 1)
            y = y.reshape(1, size, size, 1, 1)
            x = x.repeat(1, 1, 1, 3, 1)
            y = y.repeat(1, 1, 1, 3, 1)

            grid = torch.cat([x, y, anchors_i], dim=-1)
            self.register_buffer(f"grid_{size}", grid)


    def _transform_pred(self, pred, size, batch_size):
        return pred.permute(0, 2, 3, 1).reshape(batch_size, size, size, 3, -1)

    def forward(self, pred, target):
        batch_size = pred[0].shape[0]
        device = pred[0].device
        losses = []
        gt_boxes_all = target[3]

        for i in range(len(self.list_size)):
            size = pred[i].shape[2]
            anchors_i = self.anchors[3*i : 3*i+3]
            pred_i_obj = self._transform_pred(pred[i], size, batch_size)
            target_i_obj = target[i]

            pred_xy = torch.sigmoid(pred_i_obj[..., :2])
            pred_wh = pred_i_obj[..., 2:4]
            pred_boxes = torch.cat([pred_xy, pred_wh], dim=-1)
            pred_class = pred_i_obj[..., 5:]

            target_boxes = target_i_obj[..., :4]
            target_class = target_i_obj[..., 5:]

            obj_box_idx = target_i_obj[..., 4] == 1

            with torch.no_grad():
                pred_xyxy = xcell_ycell_wh2xyxy(pred_boxes, size=size, count=3, anchors=anchors_i)
                ignore_mask = torch.zeros_like(obj_box_idx, device=device)

                for b in range(batch_size):
                    gt_xyxy_b = gt_boxes_all[b].to(device)
                    if gt_xyxy_b.shape[0] == 0:
                        continue

                    iou_b = box_iou(pred_xyxy[b].reshape(-1, 4), gt_xyxy_b.reshape(-1, 4))
                    max_iou_b = iou_b.max(dim=1).values.reshape(size, size, 3)
                    ignore_mask[b] = max_iou_b > 0.5

            no_obj_box_idx = ~obj_box_idx & (~ignore_mask)

            obj_box_pred_boxes = pred_boxes[obj_box_idx]
            obj_box_target_boxes = target_boxes[obj_box_idx]

            xywh_loss = (obj_box_pred_boxes - obj_box_target_boxes).square().sum()

            pred_conf = pred_i_obj[..., 4][obj_box_idx]
            with torch.no_grad():
                target_conf = target_i_obj[..., 4][obj_box_idx]

            conf_loss = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="sum")

            class_loss = F.binary_cross_entropy_with_logits(pred_class[obj_box_idx], target_class[obj_box_idx], reduction="sum")

            no_obj_box_loss = F.binary_cross_entropy_with_logits(pred_i_obj[..., 4][no_obj_box_idx], torch.zeros_like(pred_i_obj[..., 4][no_obj_box_idx]), reduction="sum")

            n_pos = obj_box_idx.sum().clamp(min=1)

            losses.append((self.lambda_coord * xywh_loss + conf_loss + class_loss) / n_pos + (self.lambda_noobj * no_obj_box_loss) / no_obj_box_idx.sum().clamp(min=1))

        loss = sum(losses)/batch_size

        return loss, loss.item()

    @torch.no_grad()
    def forward_eval(self, pred, image_size=416):
        inference_pred = []
        batch_size = pred[0].shape[0]
        
        for i in range(len(self.list_size)):
            size = pred[i].shape[2]
            grid_i = getattr(self, f"grid_{size}")
            pred_i = self._transform_pred(pred[i], size, batch_size)
            anchors_i = self.anchors[3*i : 3*i+3]

            #pred_xyxy = xcell_ycell_wh2xyxy(pred_i, size=size, count=3, anchors=anchors_i)

            xy = (F.sigmoid(pred_i[..., :2]) + grid_i[..., :2])/size
            wh = grid_i[..., 2:4] * torch.exp(pred_i[..., 2:4]).clamp(max=10)
            conf_class = F.sigmoid(pred_i[..., 4:])

            xy = xy * image_size
            wh = wh * image_size

            inference_pred.append(torch.cat([xy, wh, conf_class], dim=-1).reshape(batch_size, size*size*3, -1))

        return torch.cat(inference_pred, dim=1)
            