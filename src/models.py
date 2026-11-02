"""Model zoo: a compact baseline CNN and a ResNet-18 adapted to small wafer maps."""
import torch.nn as nn
from torchvision.models import resnet18


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(_block(in_ch, 32), _block(32, 64), _block(64, 128), _block(128, 256))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    @property
    def cam_layer(self):
        return self.features[-1]


class WaferResNet(nn.Module):
    def __init__(self, num_classes, in_ch=3):
        super().__init__()
        net = resnet18(weights=None, num_classes=num_classes)
        # 64x64 inputs: drop the aggressive stem downsampling used for ImageNet.
        net.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        self.net = net

    def forward(self, x):
        return self.net(x)

    @property
    def cam_layer(self):
        return self.net.layer4


MODELS = {"cnn": SimpleCNN, "resnet18": WaferResNet}


def build_model(name, num_classes):
    return MODELS[name](num_classes)
