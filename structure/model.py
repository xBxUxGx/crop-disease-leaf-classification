# coding:utf-8
from torch import nn
import torchvision

# 特征图基数
nf = 16


# 自定义轻量级 CNN（3 层卷积 + 2 层全连接）
class SimpleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 8 * nf * 4, nf * 4),  # 4096 -> 64
            nn.ReLU(True),
            nn.Linear(nf * 4, num_classes),  # 64 -> 10
        )

    def forward(self, x):
        x = self.conv1(x)  # 尺寸（32，32，16）
        x = self.conv2(x)  # 尺寸（16，16，32）
        x = self.conv3(x)  # 尺寸（8，8，64）
        x = self.linear(x)  # 尺寸（10）
        return x


# ResNet18 迁移学习（预训练权重 + 替换 FC 层）
class ResNetTransfer(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetTransfer, self).__init__()
        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)  # 替换为 10 分类

    def forward(self, x):
        return self.backbone(x)
