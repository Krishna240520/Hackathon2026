"""
model.py

Builds an ImageNet-pretrained CNN with its final layer replaced for the
target number of classes. Used for both the Stage 1 species gate and the
Stage 2 breed classifiers (they share the same architecture code, just
trained on different data/num_classes).
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, arch: str = "efficientnet_b0", pretrained: bool = True):
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported arch: {arch}")
    return net


def get_input_size(arch: str) -> int:
    # All supported backbones here use 224x224 ImageNet-style input.
    return 224
