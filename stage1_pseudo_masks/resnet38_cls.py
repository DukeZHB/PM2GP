"""Multi-label classification heads on top of the ResNet-38 backbone.

``Net`` is the training model (three image-level prediction branches:
two intermediate taps ``ic1``/``ic2`` plus the final ``fc8`` head).
``Net_CAM`` additionally exposes ``forward_cam``, which returns the
three per-class CAMs used for pseudo-mask generation.

Imported dynamically by ``train_classifier.py`` via
``importlib.import_module(args.network)``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from stage1_pseudo_masks import resnet38d


class Net(resnet38d.Net):
    def __init__(self, n_class):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]
        self.ic1 = nn.Conv2d(512, n_class, 1)
        self.ic2 = nn.Conv2d(1024, n_class, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):

        x1, x2, x = super().forward(x)  # [B,512,h,w], [B,1024,h,w], [B,4096,h,w]
        x1 = self.ic1(x1)
        x2 = self.ic2(x2)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x2 = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x = self.dropout7(x)
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)

        feature = x
        feature = feature.view(feature.size(0), -1)

        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)

        return x1, x2, x, feature, y

    def forward_cam(self, x):
        x = super().forward(x)
        x = F.conv2d(x, self.fc8.weight)
        x = F.relu(x)

        return x

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Net_CAM(resnet38d.Net):
    def __init__(self, n_class):
        super().__init__()
        self.dropout7 = torch.nn.Dropout2d(0.5)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc8 = nn.Conv2d(4096, n_class, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8]
        self.ic1 = nn.Conv2d(512, n_class, 1)
        self.ic2 = nn.Conv2d(1024, n_class, 1)

    def forward(self, x):
        x = super().forward(x)
        x = self.dropout7(x)
        x = self.pool(x)
        x = self.fc8(x)
        x = x.view(x.size(0), -1)
        y = torch.sigmoid(x)
        return y

    def forward_cam(self, x):
        x1, x2, x_ = super().forward(x)
        # CAM from the b4_5 tap.
        x1 = F.conv2d(x1, self.ic1.weight)
        cam_x1 = F.relu(x1)

        # CAM from the b5_2 tap.
        x2 = F.conv2d(x2, self.ic2.weight)
        cam_x2 = F.relu(x2)

        x_pool = F.avg_pool2d(x_, kernel_size=(x_.size(2), x_.size(3)), padding=0)
        x = F.conv2d(x_, self.fc8.weight)
        cam = F.relu(x)
        y = self.fc8(x_pool)
        y = y.view(y.size(0), -1)
        y = torch.sigmoid(y)

        return cam_x1, cam_x2, cam, y

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
