import torch
import torch.nn as nn

class RadarDetectionModel(nn.Module):
    def __init__(self, backbone, detection_head):
        super(RadarDetectionModel, self).__init__()
        self.backbone = backbone
        self.detection_head = detection_head

    def forward(self, x):
        features = self.backbone(x)
        predictions = self.detection_head(features)
        return predictions