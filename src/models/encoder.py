import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet50Encoder(nn.Module):
    def __init__(self, out_dim=2048, pretrained=True):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # отбрасываем финальную классифик. голову
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # до avgpool -> [B, 2048, 1, 1]
        self.out_dim = out_dim
        # проекция не обязательна (2048 уже ок); оставляю, чтобы можно было сжать до 1024
        self.proj = nn.Identity()

    def forward(self, x):
        feats = self.backbone(x)           # [B, 2048, 1, 1]
        feats = feats.view(feats.size(0), -1)  # [B, 2048]
        feats = self.proj(feats)               # [B, D]
        feats = F.normalize(feats, p=2, dim=-1)
        return feats
