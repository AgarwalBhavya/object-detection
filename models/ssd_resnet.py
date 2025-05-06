import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.ssd import SSDHead, SSD300_VGG16_Weights, ssd300_vgg16

def get_ssd_resnet_model(num_classes):
    # Load a pretrained SSD model and replace the backbone
    backbone = models.resnet50(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool and fc

    # Feature map outputs for SSD (reduce resolution)
    class ResNetBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
        def forward(self, x):
            x = self.backbone(x)
            return {"0": x}

    model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
    model.backbone = ResNetBackbone()

    in_channels = [2048]  # Output channels of ResNet-50
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head = SSDHead(in_channels=in_channels, num_anchors=num_anchors, num_classes=num_classes)
    return model
