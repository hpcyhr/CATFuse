"""
Spiking AlexNet — SJ-compatible implementation for CATFuse benchmarks.

Architecture follows the standard AlexNet with SJ layer wrappers:
  Conv(3,64,11,s=4,p=2) → BN → LIF → MaxPool
  Conv(64,192,5,p=2) → BN → LIF → MaxPool
  Conv(192,384,3,p=1) → BN → LIF
  Conv(384,256,3,p=1) → BN → LIF
  Conv(256,256,3,p=1) → BN → LIF → MaxPool
  Linear(256*6*6,4096) → LIF → Dropout
  Linear(4096,4096) → LIF → Dropout
  Linear(4096,num_classes)

Input: 224×224 (ImageNet scale) or 32×32 (CIFAR, with adjusted pooling)
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional


class SpikingAlexNet(nn.Module):
    def __init__(self, num_classes=10, spiking_neuron=None, input_size=32, **kwargs):
        super().__init__()
        if spiking_neuron is None:
            spiking_neuron = neuron.LIFNode

        # Extract neuron kwargs
        nk = {k: v for k, v in kwargs.items()
              if k in ('tau', 'v_threshold', 'v_reset', 'surrogate_function', 'detach_reset')}

        if input_size == 224:
            # ImageNet: standard AlexNet
            self.features = nn.Sequential(
                layer.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                layer.BatchNorm2d(64),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=3, stride=2),

                layer.Conv2d(64, 192, kernel_size=5, padding=2),
                layer.BatchNorm2d(192),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=3, stride=2),

                layer.Conv2d(192, 384, kernel_size=3, padding=1),
                layer.BatchNorm2d(384),
                spiking_neuron(**nk),

                layer.Conv2d(384, 256, kernel_size=3, padding=1),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),

                layer.Conv2d(256, 256, kernel_size=3, padding=1),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=3, stride=2),
            )
            fc_in = 256 * 6 * 6
        else:
            # CIFAR-10 (32×32): adjusted strides and pooling
            self.features = nn.Sequential(
                layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                layer.BatchNorm2d(64),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=2, stride=2),  # 16×16

                layer.Conv2d(64, 192, kernel_size=3, padding=1),
                layer.BatchNorm2d(192),
                spiking_neuron(**nk),
                layer.MaxPool2d(kernel_size=2, stride=2),  # 8×8

                layer.Conv2d(192, 384, kernel_size=3, padding=1),
                layer.BatchNorm2d(384),
                spiking_neuron(**nk),

                layer.Conv2d(384, 256, kernel_size=3, padding=1),
                layer.BatchNorm2d(256),
                spiking_neuron(**nk),

                layer.Conv2d(256, 256, kernel_size=3, padding=1),
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
