import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Union


class SIoULoss(nn.Module):
    """Scylla IoU Loss combining angle, distance, shape, and IoU costs"""
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: [N, 4] (x_center, y_center, width, height)
            target_boxes: [N, 4] (x_center, y_center, width, height)
        """
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
        
        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2
        
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + self.eps
        
        iou = inter_area / union_area
        
        cw = torch.max(pred_x2, target_x2) - torch.min(pred_x1, target_x1)
        ch = torch.max(pred_y2, target_y2) - torch.min(pred_y1, target_y1)
        
        s_cw = (target_x1 + target_x2 - pred_x1 - pred_x2) * 0.5
        s_ch = (target_y1 + target_y2 - pred_y1 - pred_y2) * 0.5
        
        sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + self.eps)
        sin_alpha_1 = torch.abs(s_cw) / sigma
        sin_alpha_2 = torch.abs(s_ch) / sigma
        threshold = np.sqrt(2) / 2
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        angle_cost = 1 - 2 * torch.sin(torch.arcsin(sin_alpha) - np.pi / 4) ** 2
        
        rho_x = (s_cw / (cw + self.eps)) ** 2
        rho_y = (s_ch / (ch + self.eps)) ** 2
        gamma = 2 - angle_cost
        distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
        
        omega_w = torch.abs(pred_boxes[:, 2] - target_boxes[:, 2]) / torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        omega_h = torch.abs(pred_boxes[:, 3] - target_boxes[:, 3]) / torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        shape_cost = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
        
        siou = iou - 0.5 * (distance_cost + shape_cost)
        return 1 - siou


class FocalLoss(nn.Module):
    def __init__(self, alpha: Union[float, torch.Tensor]=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Can be scalar or tensor now
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                # Single alpha (current behavior)
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # Per-class alpha (your use case)
                alpha_t = self.alpha[None, :] * targets  # Broadcast weights
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class YOLOLoss(nn.Module):
    """Complete YOLO loss with SIoU for boxes and Focal Loss for classification"""
    
    def __init__(
        self,
        num_classes: int,
        anchors: List[List[float]] = None,
        img_size: int = 640,
        box_weight: float = 0.05,
        obj_weight: float = 1.0,
        cls_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.na = 3  # anchors per scale
        
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],      # P3/8
                [[30, 61], [62, 45], [59, 119]],     # P4/16
                [[116, 90], [156, 198], [373, 326]]  # P5/32
            ])
        else:
            self.anchors = torch.tensor(anchors)
        
        self.strides = torch.tensor([8, 16, 32])
        
        self.siou_loss = SIoULoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            predictions: List of [B, A*(5+C), H, W] for each scale
            targets: [N, 6] (batch_idx, class, x, y, w, h) normalized 0-1
        Returns:
            total_loss, (box_loss, obj_loss, cls_loss)
        """
        device = predictions[0].device
        self.anchors = self.anchors.to(device)
        self.strides = self.strides.to(device)
        
        batch_size = predictions[0].shape[0]
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        
        tcls, tbox, indices, anchors = self._build_targets(predictions, targets)
        
        for i, pred in enumerate(predictions):
            b, _, h, w = pred.shape
            pred = pred.view(b, self.na, 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
            
            pred_obj = pred[..., 4]
            target_obj = torch.zeros_like(pred_obj)
            
            n = len(indices[i][0])
            if n:
                b_idx, a_idx, gj, gi = indices[i]
                
                pred_xy = pred[b_idx, a_idx, gj, gi, 0:2].sigmoid()
                pred_wh = pred[b_idx, a_idx, gj, gi, 2:4].sigmoid() ** 2 * anchors[i]
                pred_cls = pred[b_idx, a_idx, gj, gi, 5:]
                
                pred_boxes = torch.cat([pred_xy, pred_wh], dim=1)
                siou = self.siou_loss(pred_boxes, tbox[i])
                box_loss += siou.mean()
                
                with torch.no_grad():
                    iou = self._calculate_iou(pred_boxes, tbox[i])
                    target_obj[b_idx, a_idx, gj, gi] = iou.clamp(0).type(target_obj.dtype)
                
                if self.num_classes > 1:
                    target_cls = torch.zeros_like(pred_cls)
                    target_cls[range(n), tcls[i]] = 1.0
                    cls_loss += self.focal_loss(pred_cls, target_cls)
            
            obj_loss += self.bce_obj(pred_obj, target_obj).mean()
        
        box_loss *= self.box_weight
        obj_loss *= self.obj_weight
        cls_loss *= self.cls_weight
        
        total_loss = box_loss + obj_loss + cls_loss
        loss_items = torch.stack([box_loss, obj_loss, cls_loss]).detach()
        
        return total_loss, loss_items
    
    def _build_targets(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[List, List, List, List]:
        """Build targets for each prediction scale"""
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        
        ai = torch.arange(self.na, device=targets.device).float().view(self.na, 1).repeat(1, nt)
        targets = torch.cat([targets.repeat(self.na, 1, 1), ai[:, :, None]], 2)
        
        g = 0.5
        off = torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
        ], device=targets.device).float() * g
        
        for i, pred in enumerate(predictions):
            anchors = self.anchors[i] / self.strides[i]
            gain[2:6] = torch.tensor(pred.shape)[[3, 2, 3, 2]]
            
            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < 4.0
                t = t[j]
                
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp(0, int(gain[3]) - 1), gi.clamp(0, int(gain[2]) - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indices, anch
    
    @staticmethod
    def _calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between boxes in xywh format"""
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
        
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + 1e-7
        
        return inter_area / union_area


class ModelEMA:
    """Exponential Moving Average for model weights"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
        self.updates = 0
    
    def update(self, model: nn.Module):
        self.updates += 1
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000)))
        
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.ema[k] *= d
                    self.ema[k] += (1 - d) * v.detach()
    
    def apply_to(self, model: nn.Module):
        model.load_state_dict(self.ema)


if __name__ == '__main__':
    print("Testing YOLO Loss Function\n" + "=" * 60)
    
    criterion = YOLOLoss(num_classes=80, img_size=640)
    
    predictions = [
        torch.randn(2, 255, 80, 80),
        torch.randn(2, 255, 40, 40),
        torch.randn(2, 255, 20, 20),
    ]
    
    targets = torch.tensor([
        [0, 10, 0.5, 0.5, 0.2, 0.3],
        [0, 25, 0.3, 0.7, 0.15, 0.25],
        [1, 5, 0.6, 0.4, 0.25, 0.35],
    ])
    
    total_loss, (box_loss, obj_loss, cls_loss) = criterion(predictions, targets)
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Box Loss (SIoU): {box_loss.item():.4f}")
    print(f"Objectness Loss: {obj_loss.item():.4f}")
    print(f"Classification Loss (Focal): {cls_loss.item():.4f}")