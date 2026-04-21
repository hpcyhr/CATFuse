"""
Spiking ZFNet — SJ-compatible implementation for CATFuse benchmarks.

ZFNet is a variant of AlexNet with:
  - First conv: 7×7 stride=2 (instead of 11×11 stride=4)
  - Rest is identical to AlexNet

Input: 224×224 (ImageNet) or 32×32 (CIFAR)
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional


class SpikingZFNet(nn.Module):
    def __init__(self, num_classes=10, spiking_neuron=None, input_size=32, **kwargs):
        super().__init__()
        if spiking_neuron is None:
            spiking_neuron = neuron.LIFNode

        nk = {k: v for k, v in kwargs.items()
              if k in ('tau', 'v_threshold', 'v_reset', 'surrogate_function', 'detach_reset')}

        if input_size == 224:
            # ImageNet: ZFNet first conv is 7×7 s=2 p=1
            self.features = nn.Sequential(
                layer.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
                layer.BatchNorm2d(96),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=3, stride=2),

                layer.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=3, stride=2),

                layer.Conv2d(256, 384, kernel_size=3, padding=1),
                layer.BatchNorm2d(384),
                spiking_neuron(**nk),

                layer.Conv2d(384, 384, kernel_size=3, padding=1),
                layer.BatchNorm2d(384),
                spiking_neuron(**nk),

                layer.Conv2d(384, 256, kernel_size=3, padding=1),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=3, stride=2),
            )
            fc_in = 256 * 5 * 5
        else:
            # CIFAR-10 (32×32): adjusted
            self.features = nn.Sequential(
                layer.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                layer.BatchNorm2d(96),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=2, stride=2),  # 16×16

                layer.Conv2d(96, 256, kernel_size=3, padding=1),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=2, stride=2),  # 8×8

                layer.Conv2d(256, 384, kernel_size=3, padding=1),
                layer.BatchNorm2d(384),
                spiking_neuron(**nk),

                layer.Conv2d(384, 384, kernel_size=3, padding=1),
                layer.BatchNorm2d(384),
                spiking_neuron(**nk),

                layer.Conv2d(384, 256, kernel_size=3, padding=1),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=2, stride=2),  # 4×4
            )
            fc_in = 256 * 4 * 4

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(fc_in, 4096),
            spiking_neuron(**nk),
            layer.Dropout(0.5),
            layer.Linear(4096, 4096),
            spiking_neuron(**nk),
            layer.Dropout(0.5),
            layer.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
