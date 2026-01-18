import torch
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict


class YOLOMetrics:
    """Evaluation metrics for object detection"""
    
    def __init__(self, num_classes: int, iou_thresholds: List[float] = None):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.8 for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        self.stats = []
    
    def update(self, predictions: List[torch.Tensor], targets: torch.Tensor):
        """
        Args:
            predictions: List of [N, 6] tensors (x1, y1, x2, y2, conf, class)
            targets: [M, 6] tensor (batch_idx, class, x, y, w, h) normalized
        """
        for batch_idx, pred in enumerate(predictions):
            target = targets[targets[:, 0] == batch_idx][:, 1:]
            
            if len(target) == 0:
                if len(pred) == 0:
                    continue
                self.stats.append((
                    torch.zeros(0, dtype=torch.bool),
                    pred[:, 4].cpu(),
                    pred[:, 5].cpu(),
                    torch.zeros(0, dtype=torch.long)
                ))
                continue
            
            if len(pred) == 0:
                self.stats.append((
                    torch.zeros(0, dtype=torch.bool),
                    torch.zeros(0),
                    torch.zeros(0),
                    target[:, 0].cpu()
                ))
                continue
            
            correct = self._match_predictions(pred, target)
            self.stats.append((
                correct,
                pred[:, 4].cpu(),
                pred[:, 5].cpu(),
                target[:, 0].cpu()
            ))
    
    def _match_predictions(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match predictions to targets across IoU thresholds"""
        correct = torch.zeros(len(pred), len(self.iou_thresholds), dtype=torch.bool)
        
        iou = self._box_iou(self._xywh2xyxy(target[:, 1:5]), pred[:, :4])
        
        for i, threshold in enumerate(self.iou_thresholds):
            matches = []
            for pred_idx in range(len(pred)):
                gt_match = -1
                max_iou = threshold
                
                for gt_idx in range(len(target)):
                    if iou[gt_idx, pred_idx] > max_iou and target[gt_idx, 0] == pred[pred_idx, 5]:
                        if gt_idx not in [m[1] for m in matches]:
                            max_iou = iou[gt_idx, pred_idx]
                            gt_match = gt_idx
                
                if gt_match >= 0:
                    matches.append((pred_idx, gt_match))
                    correct[pred_idx, i] = True
        
        return correct
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes"""
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        union_area = area1[:, None] + area2 - inter_area
        
        return inter_area / union_area
    
    @staticmethod
    def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Convert from center format to corner format"""
        xy = boxes[:, :2]
        wh = boxes[:, 2:4]
        return torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        if not self.stats:
            return {
                'mAP@0.5': 0.0,
                'mAP@0.5:0.95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        
        if len(stats) and stats[0].any():
            ap_per_class, p, r = self._compute_ap(*stats)
            ap50 = ap_per_class[:, self.iou_thresholds.index(0.5)]
            ap = ap_per_class.mean(1)
            
            mp = p.mean() if len(p) else 0.0
            mr = r.mean() if len(r) else 0.0
            mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
            
            return {
                'mAP@0.5': ap50.mean(),
                'mAP@0.5:0.95': ap.mean(),
                'precision': mp,
                'recall': mr,
                'f1': mf1,
                'ap_per_class': ap
            }
        
        return {
            'mAP@0.5': 0.0,
            'mAP@0.5:0.95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    def _compute_ap(self, correct, conf, pred_cls, target_cls):
        """Compute average precision per class"""
        i = np.argsort(-conf)
        correct, conf, pred_cls = correct[i], conf[i], pred_cls[i]
        
        unique_classes = np.unique(target_cls)
        n_classes = unique_classes.shape[0]
        
        ap = np.zeros((n_classes, len(self.iou_thresholds)))
        p = np.zeros(n_classes)
        r = np.zeros(n_classes)
        
        for ci, c in enumerate(unique_classes):
            i_class = pred_cls == c
            n_gt = (target_cls == c).sum()
            n_pred = i_class.sum()
            
            if n_pred == 0 or n_gt == 0:
                continue
            
            fpc = (1 - correct[i_class]).cumsum(0)
            tpc = correct[i_class].cumsum(0)
            
            recall = tpc / (n_gt + 1e-16)
            precision = tpc / (tpc + fpc)
            
            for ti, threshold in enumerate(self.iou_thresholds):
                ap[ci, ti] = self._compute_ap_single(recall[:, ti], precision[:, ti])
            
            p[ci] = precision[:, 0].mean()
            r[ci] = recall[:, 0].mean()
        
        return ap, p, r
    
    @staticmethod
    def _compute_ap_single(recall, precision):
        """Compute AP for single IoU threshold using 101-point interpolation"""
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return ap


class ConfusionMatrix:
    """Confusion matrix for object detection"""
    
    def __init__(self, num_classes: int, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
    
    def update(self, predictions: List[torch.Tensor], targets: torch.Tensor):
        """Update confusion matrix with batch predictions"""
        for batch_idx, pred in enumerate(predictions):
            target = targets[targets[:, 0] == batch_idx][:, 1:]
            
            pred = pred[pred[:, 4] > self.conf_threshold]
            
            if len(pred) == 0:
                if len(target) > 0:
                    for tc in target[:, 0].long():
                        self.matrix[tc, self.num_classes] += 1
                continue
            
            if len(target) == 0:
                for pc in pred[:, 5].long():
                    self.matrix[self.num_classes, pc] += 1
                continue
            
            target_boxes = self._xywh2xyxy(target[:, 1:5])
            iou = self._box_iou(target_boxes, pred[:, :4])
            
            matches = []
            for i, pred_box in enumerate(pred):
                pred_class = int(pred_box[5])
                
                best_iou = self.iou_threshold
                best_gt = -1
                
                for j, target_box in enumerate(target):
                    if j in matches:
                        continue
                    
                    if iou[j, i] > best_iou:
                        best_iou = iou[j, i]
                        best_gt = j
                
                if best_gt >= 0:
                    target_class = int(target[best_gt, 0])
                    self.matrix[target_class, pred_class] += 1
                    matches.append(best_gt)
                else:
                    self.matrix[self.num_classes, pred_class] += 1
            
            for j, target_box in enumerate(target):
                if j not in matches:
                    target_class = int(target_box[0])
                    self.matrix[target_class, self.num_classes] += 1
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        
        inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
        inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
        inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
        inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
        
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        union_area = area1[:, None] + area2 - inter_area
        
        return inter_area / union_area
    
    @staticmethod
    def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
        xy = boxes[:, :2]
        wh = boxes[:, 2:4]
        return torch.cat([xy - wh / 2, xy + wh / 2], dim=1)
    
    def get_matrix(self) -> np.ndarray:
        return self.matrix
    
    def plot(self, class_names: List[str] = None, save_path: str = None):
        """Plot confusion matrix"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Install matplotlib and seaborn to plot confusion matrix")
            return
        
        matrix = self.matrix.copy()
        matrix = matrix / (matrix.sum(1, keepdims=True) + 1e-16)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.num_classes)]
        class_names = class_names + ["Background"]
        
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()


class SpeedMetrics:
    """Track inference speed metrics"""
    
    def __init__(self):
        self.times = []
    
    def update(self, time: float):
        self.times.append(time)
    
    def compute(self) -> Dict[str, float]:
        if not self.times:
            return {'fps': 0.0, 'latency_ms': 0.0}
        
        times = np.array(self.times)
        mean_time = times.mean()
        
        return {
            'fps': 1.0 / mean_time if mean_time > 0 else 0.0,
            'latency_ms': mean_time * 1000,
            'latency_std_ms': times.std() * 1000
        }


if __name__ == '__main__':
    print("Testing YOLO Metrics\n" + "=" * 60)
    
    num_classes = 80
    metrics = YOLOMetrics(num_classes=num_classes)
    confusion = ConfusionMatrix(num_classes=num_classes)
    speed = SpeedMetrics()
    
    predictions = [
        torch.tensor([
            [100, 100, 200, 200, 0.9, 10],
            [300, 300, 400, 400, 0.85, 25],
        ]),
        torch.tensor([
            [150, 150, 250, 250, 0.7, 5],
        ])
    ]
    
    targets = torch.tensor([
        [0, 10, 0.375, 0.375, 0.15625, 0.15625],
        [0, 25, 0.875, 0.875, 0.15625, 0.15625],
        [1, 5, 0.3125, 0.3125, 0.15625, 0.15625],
    ])
    
    metrics.update(predictions, targets)
    confusion.update(predictions, targets)
    speed.update(0.015)
    speed.update(0.016)
    speed.update(0.014)
    
    results = metrics.compute()
    print(f"mAP@0.5: {results['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95: {results['mAP@0.5:0.95']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    
    speed_results = speed.compute()
    print(f"\nFPS: {speed_results['fps']:.1f}")
    print(f"Latency: {speed_results['latency_ms']:.2f} ± {speed_results['latency_std_ms']:.2f} ms")
    
    print("\nConfusion Matrix shape:", confusion.get_matrix().shape)
    print("=" * 60)