# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

from thop import profile

import torch
import torch.nn as nn

from src.model.model import ECG_model
from src.utils import Print


def get_model(model_cfg, num_channels, num_classes):
    """ get model supporting different model types """
    model = ECG_model(model_cfg, num_channels, num_classes)
    params = get_params_and_initialize(model)

    return model, params


def get_params_and_initialize(model):
    """
    parameter initialization
    get weights and biases for different weighty decay during training
    """
    params_with_decay, params_without_decay = [], []
    for name, param in model.named_parameters():
        if "weight" in name:
            if "bn" not in name:
                nn.init.kaiming_normal_(param, nonlinearity='relu')
                params_with_decay.append(param)
            else:
                nn.init.ones_(param)
                params_without_decay.append(param)

        else:
            nn.init.zeros_(param)
            params_without_decay.append(param)

    return params_with_decay, params_without_decay


def get_profile(model, num_channels, chunk_length, output=None):
    """ get Params, FLOPs (in # of multiply-adds) """
    input = torch.randn(1, num_channels, chunk_length)
    flags = torch.ones((1), dtype=torch.bool)
    macs, params = profile(model, inputs=(input, flags), verbose=False)

    if output is not None:
        Print("Params(M): %.3f" % (params / 10**6), output)
        Print("FLOPs(G): %.3f" % (macs / 10**9), output)
