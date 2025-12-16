import torch
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional
from tqdm import tqdm


def find_optimal_anchors(
    dataloader,
    num_anchors: int = 9,
    img_size: int = 640,
    max_samples: int = 10000,
    total: Optional[int] = None,
    verbose: bool = True
) -> List[List[List[float]]]:
    """
    Find optimal anchor boxes using K-means clustering on dataset.
    
    Args:
        dataloader: DataLoader with targets format [batch_idx, class, x, y, w, h]
        num_anchors: Total number of anchors (default 9 for 3 scales × 3 anchors)
        img_size: Input image size (default 640)
        max_samples: Max number of boxes to use (memory limit)
        verbose: Print progress
    
    Returns:
        List of anchors grouped by scale [[small], [medium], [large]]
    """
    boxes = []
    sample_count = 0
    
    if verbose:
        print(f"Collecting ground truth boxes (max {max_samples})...")
    
    pbar = tqdm(dataloader, desc="Processing batches") if verbose else dataloader
    
    for batch in pbar:
        if isinstance(batch, tuple):
            _, targets = batch
        else:
            targets = batch
        
        if targets.numel() == 0:
            continue
        
        # Extract width and height (normalized 0-1)
        widths = targets[:, 4].cpu().numpy() * img_size
        heights = targets[:, 5].cpu().numpy() * img_size
        
        # Stack and append
        batch_boxes = np.stack([widths, heights], axis=1)
        boxes.append(batch_boxes)
        
        sample_count += len(batch_boxes)
        
        if sample_count >= max_samples:
            if verbose:
                print(f"\nReached {max_samples} samples, stopping collection")
            break
    
    if not boxes:
        raise ValueError("No boxes found in dataset")
    
    # Concatenate all boxes
    boxes = np.concatenate(boxes, axis=0)
    
    if len(boxes) > max_samples:
        indices = np.random.choice(len(boxes), max_samples, replace=False)
        boxes = boxes[indices]
    
    if verbose:
        print(f"\nTotal boxes collected: {len(boxes)}")
        print(f"Box size range: W=[{boxes[:, 0].min():.1f}, {boxes[:, 0].max():.1f}], "
              f"H=[{boxes[:, 1].min():.1f}, {boxes[:, 1].max():.1f}]")
    
    # Run K-means clustering
    if verbose:
        print(f"\nRunning K-means with {num_anchors} clusters...")
    
    kmeans = KMeans(
        n_clusters=num_anchors,
        random_state=42,
        n_init=10,
        max_iter=300,
        verbose=1 if verbose else 0
    )
    kmeans.fit(boxes)
    anchors = kmeans.cluster_centers_
    
    # Sort by area (width × height)
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]
    
    # Group into scales (assuming 3 anchors per scale)
    anchors_per_scale = num_anchors // 3
    small_anchors = anchors[0:anchors_per_scale].tolist()
    medium_anchors = anchors[anchors_per_scale:2*anchors_per_scale].tolist()
    large_anchors = anchors[2*anchors_per_scale:].tolist()
    
    if verbose:
        print("\nOptimal anchors found:")
        print(f"Small objects (P3/8):  {[[int(w), int(h)] for w, h in small_anchors]}")
        print(f"Medium objects (P4/16): {[[int(w), int(h)] for w, h in medium_anchors]}")
        print(f"Large objects (P5/32):  {[[int(w), int(h)] for w, h in large_anchors]}")
        
        # Calculate average IoU
        avg_iou = compute_anchor_iou(boxes, anchors)
        print(f"\nAverage IoU with dataset boxes: {avg_iou:.4f}")
        print("(Higher is better, >0.6 is good)")
    
    return [small_anchors, medium_anchors, large_anchors]


def compute_anchor_iou(boxes: np.ndarray, anchors: np.ndarray) -> float:
    """Compute average best IoU between boxes and anchors"""
    ious = []
    
    for box in boxes:
        box_area = box[0] * box[1]
        best_iou = 0
        
        for anchor in anchors:
            anchor_area = anchor[0] * anchor[1]
            
            # Intersection (assuming centered boxes)
            inter_w = min(box[0], anchor[0])
            inter_h = min(box[1], anchor[1])
            inter_area = inter_w * inter_h
            
            # Union
            union_area = box_area + anchor_area - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            best_iou = max(best_iou, iou)
        
        ious.append(best_iou)
    
    return np.mean(ious)