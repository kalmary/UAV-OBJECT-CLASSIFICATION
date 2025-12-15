import torch
import torch.nn as nn
import json
import pathlib as pth
from typing import List, Tuple, Dict, Optional, Union


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and activation"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act: str = 'silu'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act == 'silu' else nn.LeakyReLU(0.1)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial Block - core building block"""
    def __init__(self, in_ch: int, out_ch: int, n: int = 1, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * e)
        self.conv1 = ConvBlock(in_ch, hidden_ch, 1, 1, 0)
        self.conv2 = ConvBlock(in_ch, hidden_ch, 1, 1, 0)
        self.conv3 = ConvBlock(2 * hidden_ch, out_ch, 1, 1, 0)
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_ch, hidden_ch, shortcut, e=1.0) for _ in range(n)
        ])
    
    def forward(self, x):
        y1 = self.bottlenecks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class Bottleneck(nn.Module):
    """Standard bottleneck"""
    def __init__(self, in_ch: int, out_ch: int, shortcut: bool = True, e: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * e)
        self.conv1 = ConvBlock(in_ch, hidden_ch, 1, 1, 0)
        self.conv2 = ConvBlock(hidden_ch, out_ch, 3, 1, 1)
        self.add = shortcut and in_ch == out_ch
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.conv1 = ConvBlock(in_ch, hidden_ch, 1, 1, 0)
        self.conv2 = ConvBlock(hidden_ch * 4, out_ch, 1, 1, 0)
        self.pool = nn.MaxPool2d(k, 1, k // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class DetectionHead(nn.Module):
    """YOLO detection head"""
    def __init__(self, in_ch: int, num_classes: int, anchors_per_scale: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # x, y, w, h, obj, classes
        self.conv = nn.Conv2d(in_ch, anchors_per_scale * self.num_outputs, 1)
    
    def forward(self, x):
        return self.conv(x)


class YOLOBackbone(nn.Module):
    """Configurable YOLO backbone"""
    def __init__(self, config: Dict):
        super().__init__()
        depth_mult = config['depth_multiple']
        width_mult = config['width_multiple']
        
        def make_divisible(x, divisor=8):
            return int((x + divisor / 2) // divisor * divisor)
        
        def get_depth(n):
            return max(round(n * depth_mult), 1) if n > 1 else n
        
        def get_width(c):
            return make_divisible(c * width_mult)
        
        # Stem
        self.stem = ConvBlock(3, get_width(64), 6, 2, 2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBlock(get_width(64), get_width(128), 3, 2, 1),
            CSPBlock(get_width(128), get_width(128), get_depth(3))
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock(get_width(128), get_width(256), 3, 2, 1),
            CSPBlock(get_width(256), get_width(256), get_depth(6))
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock(get_width(256), get_width(512), 3, 2, 1),
            CSPBlock(get_width(512), get_width(512), get_depth(9))
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock(get_width(512), get_width(1024), 3, 2, 1),
            CSPBlock(get_width(1024), get_width(1024), get_depth(3)),
            SPPF(get_width(1024), get_width(1024))
        )
        
        self.out_channels = [get_width(256), get_width(512), get_width(1024)]
    
    def forward(self, x):
        x = self.stem(x)
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        c6 = self.stage4(c5)
        return c4, c5, c6  # Return multi-scale features


class YOLONeck(nn.Module):
    """Feature Pyramid Network neck"""
    def __init__(self, in_channels: List[int], depth_mult: float = 1.0):
        super().__init__()
        c4, c5, c6 = in_channels
        
        def get_depth(n):
            return max(round(n * depth_mult), 1) if n > 1 else n
        
        # Top-down path
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp1 = CSPBlock(c6 + c5, c5, get_depth(3), shortcut=False)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.csp2 = CSPBlock(c5 + c4, c4, get_depth(3), shortcut=False)
        
        # Bottom-up path
        self.down1 = ConvBlock(c4, c4, 3, 2, 1)
        self.csp3 = CSPBlock(c4 + c5, c5, get_depth(3), shortcut=False)
        
        self.down2 = ConvBlock(c5, c5, 3, 2, 1)
        self.csp4 = CSPBlock(c5 + c6, c6, get_depth(3), shortcut=False)
    
    def forward(self, features):
        c4, c5, c6 = features
        
        # Top-down
        p6 = c6
        p5 = self.csp1(torch.cat([self.up1(p6), c5], dim=1))
        p4 = self.csp2(torch.cat([self.up2(p5), c4], dim=1))
        
        # Bottom-up
        n4 = p4
        n5 = self.csp3(torch.cat([self.down1(n4), p5], dim=1))
        n6 = self.csp4(torch.cat([self.down2(n5), p6], dim=1))
        
        return n4, n5, n6


class YOLO(nn.Module):
    """Complete YOLO model"""
    def __init__(self, config_name: str, num_classes: int = 80):
        super().__init__()

        CONFIGS = {
            'yolo-n': create_model_config('nano', 0.33, 0.25),      # Fastest, smallest
            'yolo-s': create_model_config('small', 0.33, 0.50),     # Balanced
            'yolo-m': create_model_config('medium', 0.67, 0.75),    # Good accuracy
            'yolo-l': create_model_config('large', 1.00, 1.00),     # High accuracy
            'yolo-x': create_model_config('xlarge', 1.33, 1.25),    # Best accuracy
        }
        
        try:
            config = CONFIGS[config_name]
        except KeyError:
            raise KeyError(f"Invalid config name: {config_name}. Valid options are: {list(CONFIGS.keys())}")

        
        self.num_classes = num_classes
        
        # Build network
        self.backbone = YOLOBackbone(config)
        self.neck = YOLONeck(self.backbone.out_channels, config['depth_multiple'])
        
        # Detection heads for 3 scales
        self.heads = nn.ModuleList([
            DetectionHead(ch, self.num_classes) for ch in self.backbone.out_channels
        ])
    
    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        
        # Neck
        features = self.neck(features)
        
        # Heads
        outputs = [head(feat) for head, feat in zip(self.heads, features)]
        
        return outputs
    
    @staticmethod
    def create_model_config(name: str, depth_mult: float, width_mult: float, num_classes: int = 80):
        """Helper to create config dictionary"""
        return {
            'name': name,
            'depth_multiple': depth_mult,
            'width_multiple': width_mult,
            'num_classes': num_classes
        }
    
    @staticmethod
    def fuse_conv_bn(conv, bn):
        """Fuse convolution and batch normalization layers"""
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=True
        ).requires_grad_(False)
        
        # Prepare filters
        with torch.no_grad():
            w_conv = conv.weight.clone().view(conv.out_channels, -1)
            w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
            fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
            
            # Prepare bias
            b_conv = torch.zeros(conv.out_channels, device=conv.weight.device) if conv.bias is None else conv.bias
            b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
            fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
        
        return fused


    def fuse(self):
        """Fuse Conv2d + BatchNorm for inference speedup"""
        for m in self.modules():
            if isinstance(m, ConvBlock):
                m.conv = self.fuse_conv_bn(m.conv, m.bn)
                m.bn = nn.Identity()
        return self





def create_model_config(name: str, depth_mult: float, width_mult: float, num_classes: int = 80):
    """Helper to create config dictionary"""
    return {
        'name': name,
        'depth_multiple': depth_mult,
        'width_multiple': width_mult,
        'num_classes': num_classes
    }


# Predefined configurations
CONFIGS = {
    'yolo-n': create_model_config('nano', 0.33, 0.25),      # Fastest, smallest
    'yolo-s': create_model_config('small', 0.33, 0.50),     # Balanced
    'yolo-m': create_model_config('medium', 0.67, 0.75),    # Good accuracy
    'yolo-l': create_model_config('large', 1.00, 1.00),     # High accuracy
    'yolo-x': create_model_config('xlarge', 1.33, 1.25),    # Best accuracy
}


# Example usage
if __name__ == '__main__':
    # Create a small YOLO model
    config_name = 'yolo-n' 
    model = YOLO(config_name, num_classes=80)
    model.eval()
    
    # Test forward pass
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Output shapes:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")
    
    # Fuse for inference
    model.fuse()
    print("\nModel fused for faster inference!")