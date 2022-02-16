# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# - TorchVision

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG_model(nn.Module):
    """ ecg_model template """
    def __init__(self, cfg, num_channels, num_classes):
        super(ECG_model, self).__init__()
        block = get_block(cfg)
        N = cfg.num_blocks // 4
        k = cfg.width_factor
        nGroups = [16 * k, 16 * k, 32 * k, 64 * k, 128 * k]
        self.in_channels = nGroups[0]

        self.conv1 = conv_kx1(num_channels, nGroups[0], cfg.kernel_size, stride=1)
        self.layer1 = self._make_layer(block, cfg, nGroups[1], N, stride=1,   first_block=True)
        self.layer2 = self._make_layer(block, cfg, nGroups[2], N, cfg.stride, first_block=False)
        self.layer3 = self._make_layer(block, cfg, nGroups[3], N, cfg.stride, first_block=False)
        self.layer4 = self._make_layer(block, cfg, nGroups[4], N, cfg.stride, first_block=False)
        self.bn1 = nn.BatchNorm1d(self.layer4[-1].out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.dropout_rate if cfg.dropout_rate is not None else 0)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.linear = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, cfg, out_channels, num_blocks, stride, first_block):
        layers = []
        for b in range(num_blocks):
            if b == 0: layer = block(cfg, self.in_channels, out_channels, stride,   first_block)
            else:      layer = block(cfg, self.in_channels, out_channels, stride=1, first_block=False)
            layers.append(layer)
            self.in_channels = layer.out_channels

        return nn.Sequential(*layers)

    def forward(self, x, flags, lam=None, p=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        outputs, outputs_sample = [], []
        for i in range(len(x)):
            outputs_sample.append(out[i:i+1])
            if flags[i]:
                outputs.append(self.maxpool(torch.cat(outputs_sample, dim=2)))
                outputs_sample = []
        outputs = torch.cat(outputs, dim=0)
        if lam is not None:
            outputs = lam * outputs + (1 - lam) * outputs[p]

        outputs = outputs.view(len(outputs), -1)
        outputs = self.linear(outputs)

        return outputs


def get_block(cfg):
    """ get block for ECG model """
    if cfg.block == "ResNet_Basic":           block = ResNet_Basic_Block
    elif cfg.block == "ResNet_Bottleneck":    block = ResNet_Bottleneck_Block
    elif cfg.block == "SEResNet_Basic":       block = ResNet_Basic_Block
    elif cfg.block == "SEResNet_Bottleneck":  block = ResNet_Bottleneck_Block
    elif cfg.block == "ResNeXt_Basic":        block = ResNet_Basic_Block
    elif cfg.block == "ResNeXt_Bottleneck":   block = ResNet_Bottleneck_Block
    elif cfg.block == "SEResNeXt_Basic":        block = ResNet_Basic_Block
    elif cfg.block == "SEResNeXt_Bottleneck":   block = ResNet_Bottleneck_Block
    elif cfg.block == "ResNeSt_Bottleneck":   block = ResNeSt_Bottleneck_Block


    # elif cfg.block == "ResNeXt":   block = ResNet_Bottleneck_Block
    # elif cfg.block == "SEResNet":  block = SEResNet_Block
    # elif cfg.block == "SEResNeXt": block = SEResNet_Block
    # elif cfg.block == "ResNeSt":   block = ResNeSt_Block

    return block


def conv_kx1(in_channels, out_channels, kernel_size, cardinality=1, stride=1):
    """ kx1 convolution with padding """
    layers = []
    if cardinality is None: cardinality = 1
    padding = kernel_size - stride
    padding_left = padding // 2
    padding_right = padding - padding_left
    layers.append(nn.ConstantPad1d((padding_left, padding_right), 0))
    layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, groups=cardinality, stride=stride, bias=False))
    return nn.Sequential(*layers)


def conv_1x1(in_channels, out_channels, cardinality=1):
    """ 1x1 convolution """
    return nn.Conv1d(in_channels, out_channels, kernel_size=1, groups=cardinality, stride=1, padding=0, bias=False)


class ResNet_Basic_Block(nn.Module):
    """
    ResNet Basic Block
    Supports ResNeXt if cardinality > 1
    Supports Squeeze-and-Excitation ResNetBasic Block
    -- BN-ReLU-Conv_kx1 - BN-ReLU-Conv_kx1
    -- (GlobalAvgPool - Conv_1x1-ReLU - Conv_1x1-Sigmoid)
    -- MaxPool-Conv_1x1
    """
    def __init__(self, cfg, in_channels, out_channels, stride, first_block):
        super(ResNet_Basic_Block, self).__init__()
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.conv1 = conv_kx1(in_channels,  out_channels, cfg.kernel_size, cfg.cardinality, stride)
        self.conv2 = conv_kx1(out_channels, out_channels, cfg.kernel_size, cfg.cardinality, stride=1)
        self.first_block = first_block

        if cfg.block.startswith("SE"):
            self.se = True
            se_reduction = 4
            se_channels = out_channels // se_reduction
            self.sigmoid = nn.Sigmoid()
            self.se_avgpool = nn.AdaptiveAvgPool1d(1)
            self.se_conv1 = conv_1x1(out_channels, se_channels)
            self.se_conv2 = conv_1x1(se_channels, out_channels)
        else:
            self.se = False

        shortcut = []
        if stride != 1:
            shortcut.append(nn.MaxPool1d(stride))
        if in_channels != self.out_channels and cfg.shortcut == "conv":
            shortcut.append(conv_1x1(in_channels, self.out_channels))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        if self.first_block:
            x = self.relu(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.relu(self.bn1(x))
            out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        if self.se:
            se = self.se_avgpool(out)
            se = self.relu(self.se_conv1(se))
            se = self.sigmoid(self.se_conv2(se))
            out = out * se

        x = self.shortcut(x)
        out_c, x_c = out.shape[1], x.shape[1]
        if out_c == x_c: out += x
        else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

        return out


class ResNet_Bottleneck_Block(nn.Module):
    """
    ResNet Bottleneck Block
    Supports ResNeXt if cardinality > 1
    Supports Squeeze-and-Excitation ResNetBasic Block
    -- BN-ReLU-Conv_1x1 - BN-ReLU-Conv_kx1 - BN-ReLU-Conv_1x1
    -- (GlobalAvgPool - Conv_1x1-ReLU - Conv_1x1-Sigmoid)
    -- MaxPool-Conv_1x1
    """
    def __init__(self, cfg, in_channels, channels, stride, first_block):
        super(ResNet_Bottleneck_Block, self).__init__()
        expansion = 4
        self.out_channels = channels * expansion

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.conv1 = conv_1x1(in_channels, channels)
        self.conv2 = conv_kx1(channels, channels, cfg.kernel_size, cfg.cardinality, stride)
        self.conv3 = conv_1x1(channels, self.out_channels)
        self.first_block = first_block

        if cfg.block.startswith("SE"):
            self.se = True
            se_reduction = 4
            se_channels = self.out_channels // se_reduction
            self.sigmoid = nn.Sigmoid()
            self.se_avgpool = nn.AdaptiveAvgPool1d(1)
            self.se_conv1 = conv_1x1(self.out_channels, se_channels)
            self.se_conv2 = conv_1x1(se_channels, self.out_channels)
        else:
            self.se = False

        shortcut = []
        if stride != 1:
            shortcut.append(nn.MaxPool1d(stride))
        if in_channels != self.out_channels and cfg.shortcut == "conv":
            shortcut.append(conv_1x1(in_channels, self.out_channels))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        if self.first_block:
            x = self.relu(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.relu(self.bn1(x))
            out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.se:
            se = self.se_avgpool(out)
            se = self.relu(self.se_conv1(se))
            se = self.sigmoid(self.se_conv2(se))
            out = out * se

        x = self.shortcut(x)
        out_c, x_c = out.shape[1], x.shape[1]
        if out_c == x_c: out += x
        else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

        return out


class ResNeSt_Bottleneck_Block(nn.Module):
    """
    ResNeSt Bottleneck Block
    -- Conv_1x1-BN-ReLU - Conv_kx1-BN-ReLU - Conv_1x1-BN-ReLU
    -- GlobalAvgPool - Conv_1x1-ReLU - Conv_1x1-Sigmoid
    -- AvgPool-Conv_1x1
    """
    def __init__(self, cfg, in_channels, channels, stride, first_block):
        super(ResNeSt_Bottleneck_Block, self).__init__()
        expansion = 4
        se_reduction = 4
        self.cardinality = cfg.cardinality
        self.radix = cfg.radix
        self.out_channels = channels * expansion
        se_channels = channels * cfg.radix // se_reduction

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(channels)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=cfg.dropout_rate)
        self.conv1 = conv_1x1(in_channels, channels)
        self.conv2 = conv_kx1(channels, channels * cfg.radix, cfg.kernel_size, cfg.cardinality * cfg.radix, stride)
        self.conv3 = conv_1x1(channels, self.out_channels)
        self.sa_avgpool = nn.AdaptiveAvgPool1d(1)
        self.sa_conv1 = conv_1x1(channels, se_channels, cfg.cardinality)
        self.sa_conv2 = conv_1x1(se_channels, channels * cfg.radix, cfg.cardinality)
        self.first_block = first_block

        shortcut = []
        if stride != 1:
            shortcut.append(nn.AvgPool1d(stride))
        if in_channels != self.out_channels and cfg.shortcut == "conv":
            shortcut.append(conv_1x1(in_channels, self.out_channels))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        if self.first_block:
            x = self.relu(self.bn1(x))
            out = self.conv1(x)
        else:
            out = self.relu(self.bn1(x))
            out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)

        b, c, l = out.shape
        out = torch.split(out, c // self.radix, dim=1)
        sa = self.sa_avgpool(sum(out))
        sa = self.sa_conv1(sa)
        sa = self.relu(sa)
        sa = self.sa_conv2(sa)
        sa = self.softmax(sa.view(b, self.cardinality, self.radix, -1).transpose(1,2)).reshape(b, -1, 1)
        sa = torch.split(sa, c // self.radix, dim=1)
        out = sum([s * o for (s, o) in zip(sa, out)])

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        x = self.shortcut(x)
        out_c, x_c = out.shape[1], x.shape[1]
        if out_c == x_c: out += x
        else:            out += F.pad(x, (0, 0, 0, out_c - x_c))

        return out

