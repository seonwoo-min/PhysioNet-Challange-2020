# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import json

import torch

from src.utils import Print


class DataConfig():
    def __init__(self, file=None, idx="data_config"):
        """ data configurations """
        self.idx = idx
        self.path = None
        self.datasets = None
        self.fold = None
        self.chunk_length = None
        self.chunk_stride = None

        self.num_channels = None
        self.split = None
        self.scored_classes = None
        self.equivalent_classes = None
        self.normal_class = None

        self.data = None
        self.header = None
        self.filenames = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("data-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "path":                         self.path = value
                elif key == "datasets":                     self.datasets = value
                elif key == "fold":                         self.fold = value
                elif key == "chunk_length":                 self.chunk_length = value
                elif key == "chunk_stride":                 self.chunk_stride = value
                else: sys.exit("# ERROR: invalid key [%s] in data-config file" % key)

        # load information
        if self.path is not None:
            self.split = json.load(open(self.path + "split.json", "r"))
            info = json.load(open(self.path + "info.json", "r"))
            self.num_channels = info["num_channels"]
            self.equivalent_classes = info["equivalent_classes"]
            scored_classes = info["scored_classes"]
            for class1, class2 in self.equivalent_classes.items():
                if class1 in scored_classes:
                    scored_classes[scored_classes.index(class1)] = class2
            self.scored_classes = sorted(set(scored_classes))
            self.normal_class = info["normal_class"]

    def get_config(self):
        configs = []
        configs.append(["path", self.path])
        configs.append(["datasets", self.datasets])
        if self.fold is not None: configs.append(["fold", self.fold])
        if self.num_channels is not None: configs.append(["num_channels", self.num_channels])
        if self.chunk_length is not None: configs.append(["chunk_length", self.chunk_length])
        if self.chunk_stride is not None: configs.append(["chunk_stride", self.chunk_stride])
        if self.scored_classes is not None: configs.append(["scored_classes", len(self.scored_classes)])
        return configs


class PreprocessConfig():
    def __init__(self, file=None, idx="preprocess_config"):
        """ preprocess configurations """
        self.idx = idx
        self.all_negative = None
        self.sample_rate = None
        self.filter_notch = []
        self.filter_lowpass = []
        self.filter_highpass = []
        self.filter_bandpass = []
        self.scaler = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("preprocess-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "all_negative":                   self.all_negative = value
                elif key == "sample_rate":                  self.sample_rate = value
                elif key == "filter_highpass":              self.filter_highpass = value
                elif key == "filter_lowpass":               self.filter_lowpass = value
                elif key == "filter_bandpass":              self.filter_bandpass = value
                elif key == "filter_notch":                 self.filter_notch = value
                elif key == "scaler":                       self.scaler = value
                else: sys.exit("# ERROR: invalid key [%s] in preprocess-config file" % key)

    def get_config(self):
        configs = []
        if self.all_negative is not None: configs.append(["all_negative", self.all_negative])
        configs.append(["sample_rate", self.sample_rate])
        if len(self.filter_highpass) > 0: configs.append(["filter_highpass", self.filter_highpass])
        if len(self.filter_lowpass) > 0:  configs.append(["filter_lowpass", self.filter_lowpass])
        if len(self.filter_bandpass) > 0: configs.append(["filter_bandpass", self.filter_bandpass])
        if len(self.filter_notch) > 0:    configs.append(["filter_notch", self.filter_notch])
        if self.scaler is not None:       configs.append(["scaler", self.scaler])

        return configs


class ModelConfig():
    def __init__(self, file=None, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.block = None
        self.num_blocks = None
        self.width_factor = None
        self.kernel_size = None
        self.cardinality = None
        self.radix = None
        self.stride = None
        self.shortcut = None
        self.dropout_rate = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key   == "block":
                    supported_blocks = ["ResNet_Basic", "ResNet_Bottleneck", "SEResNet_Basic", "SEResNet_Bottleneck",
                                        "ResNeXt_Basic", "ResNeXt_Bottleneck", "SEResNeXt_Basic", "SEResNeXt_Bottleneck",
                                        "ResNeSt_Bottleneck"]
                    if value in supported_blocks:          self.block = value
                    else: sys.exit("# ERROR: invalid value [%s] for key [%s] in model-config file" % (value, key))
                elif key == "num_blocks":                   self.num_blocks = value
                elif key == "width_factor":                 self.width_factor = value
                elif key == "kernel_size":                  self.kernel_size = value
                elif key == "cardinality":                  self.cardinality = value
                elif key == "radix":                        self.radix = value
                elif key == "stride":                       self.stride = value
                elif key == "shortcut":                     self.shortcut = value
                elif key == "dropout_rate":                 self.dropout_rate = value
                else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["block", self.block])
        configs.append(["num_blocks", self.num_blocks])
        if self.width_factor is not None: configs.append(["width_factor", self.width_factor])
        configs.append(["kernel_size", self.kernel_size])
        if self.cardinality is not None: configs.append(["cardinality", self.cardinality])
        if self.radix is not None: configs.append(["radix", self.radix])
        configs.append(["stride", self.stride])
        if self.shortcut is not None: configs.append(["shortcut", self.shortcut])
        if self.dropout_rate is not None: configs.append(["dropout_rate", self.dropout_rate])

        return configs


class RunConfig():
    def __init__(self, file=None, idx="run_config", eval=False, sanity_check=False):
        """ run configurations """
        self.idx = idx
        self.eval = eval
        self.batch_size_train = None
        self.batch_size_eval = None
        self.num_epochs = None
        self.optimizer = None
        self.scheduler = None
        self.learning_rate = None
        self.learning_rate_decay = None
        self.patience = None
        self.momentum = None
        self.weight_decay = None
        self.multilabel = None
        self.class_weight = None
        self.confusion_weight = None
        self.mixup = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("run-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "batch_size_train":             self.batch_size_train = value
                elif key == "batch_size_eval":              self.batch_size_eval = value
                elif key == "num_epochs":                   self.num_epochs = value
                elif key == "optimizer":
                    if value in ["sgd", "adam"]:            self.optimizer = value
                    else: sys.exit("# ERROR: invalid value [%s] for key [%s] in run-config file" % (value, key))
                elif key == "scheduler":
                    if value in ["plateau", "warmup_plateau",
                                 "cosine", "warmup_cosine"]:self.scheduler = value
                    else: sys.exit("# ERROR: invalid value [%s] for key [%s] in run-config file" % (value, key))
                elif key == "learning_rate":                self.learning_rate = value
                elif key == "learning_rate_decay":          self.learning_rate_decay = value
                elif key == "patience":                     self.patience = value
                elif key == "momentum":                     self.momentum = value
                elif key == "weight_decay":                 self.weight_decay = value
                elif key == "multi_label":                  self.multilabel = value
                elif key == "class_weight":                 self.class_weight = value
                elif key == "confusion_weight":             self.confusion_weight = value
                elif key == "mixup":
                    if value in [False, "input", "manifold"]:self.mixup = value
                    else: sys.exit("# ERROR: invalid value [%s] for key [%s] in run-config file" % (value, key))
                else: sys.exit("# ERROR: invalid key [%s] in run-config file" % key)

        if not self.multilabel and self.confusion_weight:
            sys.exit("# ERROR: multiclass classification does not support confusion_weight")

        if sanity_check:
            self.batch_size_train = 32
            self.num_epochs = 4

    def get_config(self):
        configs = []
        if not self.eval:
            configs.append(["batch_size_train", self.batch_size_train])
        configs.append(["batch_size_eval", self.batch_size_eval])
        if not self.eval:
            configs.append(["num_epochs", self.num_epochs])
            configs.append(["optimizer", self.optimizer])
            configs.append(["scheduler", self.scheduler])
            configs.append(["learning_rate", self.learning_rate])
            if self.scheduler == "plateau":
                configs.append(["learning_rate_decay", self.learning_rate_decay])
                configs.append(["patience", self.patience])
            if self.optimizer == "sgd":
                configs.append(["momentum", self.momentum])
            configs.append(["weight_decay", self.weight_decay])
            configs.append(["multilabel", self.multilabel])
            configs.append(["class_weight", self.class_weight])
            configs.append(["confusion_weight", self.confusion_weight])
            configs.append(["mixup", self.mixup])

        return configs


def print_configs(args, cfgs, device, output):
    if args["sanity_check"]: Print(" ".join(['##### SANITY_CHECK #####']), output)
    Print(" ".join(['##### arguments #####']), output)
    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for c, v in cfg.get_config():
            Print(" ".join(['-- %s: %s' % (c, v)]), output)
    if args["checkpoint"] is not None:
        Print(" ".join(['checkpoint: %s' % (args["checkpoint"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)