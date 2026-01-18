import torch
from typing import List


def non_max_suppression(
    predictions: List[torch.Tensor],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 100
) -> List[torch.Tensor]:
    """
    Perform NMS on YOLO predictions.
    
    Args:
        predictions: List of [B, A*(5+C), H, W] for each scale
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
    
    Returns:
        List of [N, 6] tensors per image (x1, y1, x2, y2, conf, class)
    """
    batch_size = predictions[0].shape[0]
    num_classes = (predictions[0].shape[1] // 3) - 5
    device = predictions[0].device
    
    # Decode predictions from all scales
    decoded = []
    for scale_idx, pred in enumerate(predictions):
        decoded.append(_decode_predictions(pred, scale_idx, num_classes))
    
    # Concatenate all scales
    all_predictions = torch.cat(decoded, dim=1)  # [B, total_boxes, 5+C]
    
    # Apply NMS per image
    output = []
    for img_idx in range(batch_size):
        pred = all_predictions[img_idx]  # [total_boxes, 5+C]
        
        # Filter by objectness
        obj_mask = pred[:, 4] > conf_thres
        pred = pred[obj_mask]
        
        if pred.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=device))
            continue
        
        # Get class scores and predictions
        class_scores = pred[:, 5:] * pred[:, 4:5]  # Multiply by objectness
        class_conf, class_pred = class_scores.max(dim=1, keepdim=True)
        
        # Filter by class confidence
        conf_mask = class_conf.squeeze() > conf_thres
        pred = pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        
        if pred.shape[0] == 0:
            output.append(torch.zeros((0, 6), device=device))
            continue
        
        # Convert box format: [x, y, w, h] -> [x1, y1, x2, y2]
        boxes = _xywh2xyxy(pred[:, :4])
        
        # Combine for NMS
        detections = torch.cat([boxes, class_conf, class_pred.float()], dim=1)
        
        # Perform NMS per class
        keep_boxes = []
        for c in class_pred.unique():
            mask = class_pred.squeeze() == c
            boxes_c = detections[mask]
            
            if boxes_c.shape[0] == 0:
                continue
            
            # NMS
            keep = _nms(boxes_c[:, :4], boxes_c[:, 4], iou_thres)
            keep_boxes.append(boxes_c[keep])
        
        if len(keep_boxes) == 0:
            output.append(torch.zeros((0, 6), device=device))
            continue
        
        # Concatenate and limit detections
        detections = torch.cat(keep_boxes, dim=0)
        if detections.shape[0] > max_det:
            detections = detections[:max_det]
        
        output.append(detections)
    
    return output


def _decode_predictions(pred: torch.Tensor, scale_idx: int, num_classes: int) -> torch.Tensor:
    """Decode raw predictions to boxes"""
    B, _, H, W = pred.shape
    device = pred.device
    stride = 8 * (2 ** scale_idx)  # 8, 16, 32
    
    # Reshape: [B, 3*(5+C), H, W] -> [B, 3, H, W, 5+C]
    pred = pred.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
    
    # Create grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).to(device).float()
    
    # Decode boxes
    xy = (pred[..., :2].sigmoid() + grid.unsqueeze(0).unsqueeze(0)) * stride
    wh = pred[..., 2:4].exp() * stride  # Simplified (should use anchors)
    conf = pred[..., 4].sigmoid()
    cls = pred[..., 5:].sigmoid()
    
    # Reshape to [B, H*W*3, 5+C]
    boxes = torch.cat([xy, wh], dim=-1).view(B, -1, 4)
    conf = conf.view(B, -1, 1)
    cls = cls.view(B, -1, num_classes)
    
    return torch.cat([boxes, conf, cls], dim=-1)


def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [x_center, y_center, w, h] to [x1, y1, x2, y2]"""
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Standard NMS implementation"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
        
        # IoU with remaining boxes
        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())
        
        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU below threshold
        idx = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)