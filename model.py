import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BotnetImageCNN(nn.Module):
    """
    CNN 2D dùng pretrained MobileNetV3-Small.
    - Input: ảnh 1x32x32
    - Nội bộ resize lên 224x224 cho phù hợp với pretrained weights.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()

        # Pretrained MobileNetV3 Small (nhẹ)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base = models.mobilenet_v3_small(weights=weights)

        # Chỉnh conv đầu tiên nhận 1 kênh thay vì 3 kênh
        first_conv = base.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        # Khởi tạo trọng số từ pretrained: trung bình theo channel
        with torch.no_grad():
            new_conv.weight[:] = first_conv.weight.mean(dim=1, keepdim=True)

        base.features[0][0] = new_conv

        # Thay classifier cuối cho đúng số lớp
        in_features = base.classifier[-1].in_features
        base.classifier[-1] = nn.Linear(in_features, n_classes)

        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 32, 32] -> resize 224x224 cho MobileNet
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.base(x)


class BotnetClassifier(nn.Module):
    """
    Wrapper giữ API cũ:
      - vẫn nhận n_features, image_size nhưng bỏ qua; dữ liệu đã là ảnh 1x32x32.
    """

    def __init__(self, base_model=None, n_features=None, image_size: int = 32, n_classes: int = 3):
        super().__init__()
        self.model = BotnetImageCNN(n_classes=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
