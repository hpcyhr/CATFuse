"""Spiking MobileNet-V1 for CATFuse benchmarks."""
import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron, functional


class _SpikingDWSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride, spiking_neuron, **nk):
        super().__init__()
        self.dw = nn.Sequential(
            layer.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            layer.BatchNorm2d(in_ch),
            spiking_neuron(**nk),
        )
        self.pw = nn.Sequential(
            layer.Conv2d(in_ch, out_ch, 1, bias=False),
            layer.BatchNorm2d(out_ch),
            spiking_neuron(**nk),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


class SpikingMobileNetV1(nn.Module):
    def __init__(self, num_classes=10, spiking_neuron=None, input_size=32, **kwargs):
        super().__init__()
        if spiking_neuron is None:
            spiking_neuron = neuron.LIFNode
        nk = {k: v for k, v in kwargs.items()
              if k in ('tau', 'v_threshold', 'v_reset', 'surrogate_function', 'detach_reset')}

        first_stride = 2 if input_size == 224 else 1
        self.head = nn.Sequential(
            layer.Conv2d(3, 32, 3, stride=first_stride, padding=1, bias=False),
            layer.BatchNorm2d(32),
            spiking_neuron(**nk),
        )

        cfg = [
            (32, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2),
            (256, 256, 1), (256, 512, 2), (512, 512, 1), (512, 512, 1),
            (512, 512, 1), (512, 512, 1), (512, 512, 1), (512, 1024, 2),
            (1024, 1024, 1),
        ]
        layers = []
        for ic, oc, s in cfg:
            layers.append(_SpikingDWSeparable(ic, oc, s, spiking_neuron, **nk))
        self.body = nn.Sequential(*layers)

        self.pool = layer.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.pool(x)
        return self.classifier(x)
