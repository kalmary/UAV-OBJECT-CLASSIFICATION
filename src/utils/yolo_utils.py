import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import pathlib as pth
import sys
from thop import profile


class YOLOLoss(nn.Module):
    """YOLO loss function with configurable components"""
    def __init__(self, num_classes: int, anchors: List[List[float]], img_size: int = 640):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = torch.tensor(anchors)
        self.img_size = img_size
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Loss weights
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
    
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor):
        """
        Args:
            predictions: List of tensors [B, A*(5+C), H, W] for each scale
            targets: Tensor [N, 6] where each row is [batch_idx, class, x, y, w, h]
        """
        device = predictions[0].device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        
        # Build targets for each scale
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # Calculate losses for each scale
        for i, pred in enumerate(predictions):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pred[..., 4])
            
            n = b.shape[0]
            if n:
                # Get predictions for matched anchors
                ps = pred[b, :, gj, gi]
                ps = ps.view(n, -1, 5 + self.num_classes)
                ps = ps[torch.arange(n), a]
                
                # Box loss (CIoU)
                pxy = ps[:, :2].sigmoid()
                pwh = ps[:, 2:4].exp() * anchors[i]
                pbox = torch.cat([pxy, pwh], dim=1)
                iou = self.bbox_iou(pbox, tbox[i], CIoU=True)
                lbox += (1.0 - iou).mean()
                
                # Objectness loss
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification loss
                if self.num_classes > 1:
                    t = torch.zeros_like(ps[:, 5:])
                    t[range(n), tcls[i]] = 1.0
                    lcls += self.bce_cls(ps[:, 5:], t).mean()
            
            # Objectness loss for all predictions
            lobj += self.bce_obj(pred[..., 4], tobj).mean()
        
        # Combine losses
        lbox *= self.lambda_box
        lobj *= self.lambda_obj
        lcls *= self.lambda_cls
        
        loss = lbox + lobj + lcls
        return loss, torch.stack([lbox, lobj, lcls]).detach()
    
    def build_targets(self, predictions, targets):
        """Build targets for each prediction scale"""
        na = 3  # Number of anchors per scale
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        
        # Prepare anchor indices
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat([targets.repeat(na, 1, 1), ai[:, :, None]], 2)
        
        for i, pred in enumerate(predictions):
            anchors = self.anchors[i * na:(i + 1) * na] / (self.img_size // pred.shape[-1])
            gain[2:6] = torch.tensor(pred.shape)[[3, 2, 3, 2]]
            
            t = targets * gain
            if nt:
                # Match targets to anchors
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < 4.0
                t = t[j]
                
                # Offsets
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < 0.5) & (gxy > 1.)).T
                l, m = ((gxi % 1. < 0.5) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + 
                          torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], 
                                      device=gxy.device).float()[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            # Append
            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat([gxy - gij, gwh], 1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indices, anch
    
    @staticmethod
    def bbox_iou(box1, box2, CIoU=False):
        """Calculate IoU between boxes"""
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + 1e-16
        
        iou = inter / union
        
        if CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            c2 = cw ** 2 + ch ** 2 + 1e-16
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            
            v = (4 / (3.14159 ** 2)) * torch.pow(
                torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
            alpha = v / (v - iou + (1 + 1e-16))
            return iou - (rho2 / c2 + v * alpha)
        
        return iou


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """Perform Non-Maximum Suppression on predictions"""
    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    
    max_wh = 7680
    max_nms = 30000
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        
        if not x.shape[0]:
            continue
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        # Batched NMS
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torch.ops.torchvision.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        
        output[xi] = x[i]
    
    return output


def xywh2xyxy(x):
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]"""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6
    }


def calculate_flops(model, img_size=640):
    """Estimate FLOPs for the model"""

    device = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size).to(device)
    flops, params = profile(model, inputs=(x,), verbose=False)
    return flops / 1e9, params / 1e6


class ModelEMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.9999):
        self.ema = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.decay = decay
        self.updates = 0
    
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000)))
        
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.ema[k] *= d
                    self.ema[k] += (1 - d) * v.detach()
    
    def apply_to(self, model):
        model.load_state_dict(self.ema)