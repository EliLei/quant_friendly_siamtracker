import torch
import torch.nn as nn


class GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights=None):
        if pred.dim() == 4:
            pred = pred.unsqueeze(0)

        pred = pred.permute(0, 1, 3, 4, 2).reshape(-1, 4) # nf x ns x x 4 x h x w
        target = target.permute(0, 1, 3, 4, 2).reshape(-1, 4) #nf x ns x 4 x h x w

        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_union = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect + 1e-7
        ious = (area_intersect) / (area_union)
        gious = ious - (ac_union - area_union) / ac_union

        losses = 1 - gious

        if weights is not None and weights.sum() > 0:
            weights = weights.permute(0, 1, 3, 4, 2).reshape(-1) # nf x ns x x 1 x h x w
            loss_mean = losses[weights>0].mean()
            ious = ious[weights>0]
        else:
            loss_mean = losses.mean()

        return loss_mean, ious



from torchvision.ops.boxes import box_area
class GIoULoss_x1y1x2y2(nn.Module):
    def __init__(self):
        super(GIoULoss_x1y1x2y2, self).__init__()


    def box_iou(self, boxes1, boxes2):
        """

        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        area1 = box_area(boxes1) # (N,)
        area2 = box_area(boxes2) # (N,)

        lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
        rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

        wh = (rb - lt).clamp(min=0)  # (N,2)
        inter = wh[:, 0] * wh[:, 1]  # (N,)

        union = area1 + area2 - inter

        iou = inter / union
        return iou, union

    def generalized_box_iou(self, boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        boxes1: (N, 4)
        boxes2: (N, 4)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        # try:
        valid1 =  boxes1[:, 2:] >= boxes1[:, :2]
        valid1 = torch.logical_and(valid1[:,0],valid1[:,1])
        valid2 = boxes2[:, 2:] >= boxes2[:, :2]
        valid2 = torch.logical_and(valid2[:, 0], valid2[:, 1])

        valid = torch.logical_and(valid1,valid2)


        iou, union = self.box_iou(boxes1, boxes2) # (N,)

        lt = torch.min(boxes1[:, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # (N,2)
        area = wh[:, 0] * wh[:, 1] # (N,)
        giou = iou - (area - union)/ area

        giou = giou*valid
        iou = iou*valid

        return giou, iou


    def giou_loss_x1y1x2y2(self, boxes1, boxes2):
        """

        :param boxes1: (N, 4) (x1,y1,x2,y2)
        :param boxes2: (N, 4) (x1,y1,x2,y2)
        :return:
        """
        giou, iou = self.generalized_box_iou(boxes1, boxes2)
        return (1 - giou).mean(), iou

    def forward(self, boxes1,boxes2):
        return self.giou_loss_x1y1x2y2(boxes1,boxes2)