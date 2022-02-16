# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
from scipy.io import loadmat

import torch

from src.preprocess import preprocess_signal, preprocess_label
from src.evaluate import load_weights


class ecg_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for ecg training and evaluation """
    def __init__(self, filenames, sample_rates, Y, preprocess_cfg):
        self.filenames = filenames
        self.sample_rates = sample_rates
        self.Y = Y
        self.preprocess_cfg = preprocess_cfg

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        x = preprocess_signal(load_data(self.filenames[i]), self.preprocess_cfg, self.sample_rates[i])
        y = self.Y[i]

        return torch.from_numpy(x), y


class ecg_dataset_for_inference(torch.utils.data.Dataset):
    """ Pytorch dataloader for pre-loaded ecg data inference """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def get_dataset_from_configs(data_cfg, preprocess_cfg, dataset_idx=None, split_idx=None, sanity_check=False):
    """ load ecg dataset from config files """
    if data_cfg.data is not None:
        x = preprocess_signal(data_cfg.data, preprocess_cfg, get_sample_rate(data_cfg.header))
        y = np.zeros((1), np.float32)  # dummy label for code compatibility
        X = [torch.from_numpy(x).float()]
        Y = [torch.from_numpy(y)]
        dataset = ecg_dataset_for_inference(X, Y)
    else:
        if data_cfg.filenames is not None: filenames_all = data_cfg.filenames
        else:                              filenames_all = get_filenames_from_split(data_cfg, dataset_idx, split_idx)
        filenames, sample_rates, Y = [], [], []
        for filename in filenames_all:
            if sanity_check and len(filenames) == 64: break
            header = load_header(filename)
            y = preprocess_label(get_labels(header), data_cfg.scored_classes, data_cfg.equivalent_classes)
            if np.sum(y) != 0 or (split_idx == "train" and preprocess_cfg.all_negative):
                filenames.append(filename)
                sample_rates.append(get_sample_rate(header))
                Y.append(torch.from_numpy(y))

        dataset = ecg_dataset(filenames, sample_rates, Y, preprocess_cfg)

    return dataset


def load_data(filename):
    """ load data from WFDB files """
    data = np.asarray(loadmat(filename + ".mat")['val'], dtype=np.float32)

    return data


def load_header(filename):
    """ load header from WFDB files """
    HEADER = open(filename + ".hea", 'r')
    header = HEADER.readlines()
    HEADER.close()

    return header


def get_labels(header):
    """ get labels from header """
    labels = []
    for line in header:
        if line.startswith('#Dx'):
            labels = [label.strip() for label in line.split(': ')[1].split(',')]

    return labels


def get_sample_rate(header):
    """ get sample frequency from header """
    sample_rate = int(header[0].strip().split()[2])

    return sample_rate


def get_filenames_from_split(data_cfg, dataset_idx, split_idx):
    """ get filenames from config file and split index """
    filenames_all = []
    for dataset in data_cfg.datasets:
        if dataset_idx not in [None, "all", dataset]: continue

        filenames = []
        if data_cfg.fold in list(range(10)):
            if split_idx == "train":
                for fold in range(10):
                    if fold != data_cfg.fold:
                        filenames += data_cfg.split[dataset]["fold%d" % fold]
            else:
                filenames += data_cfg.split[dataset]["fold%d" % data_cfg.fold]
        elif data_cfg.fold in ["sanity"]:
            filenames += data_cfg.split[dataset]["all"][:1]
        else:
            filenames += data_cfg.split[dataset][split_idx]

        filenames_all += [data_cfg.path + "/%s/%s" % (dataset, filename) for filename in filenames]

    return filenames_all


def get_eval_dataset_split_idxs(data_cfg):
    """ get dataset and split idxs for evaluation """
    dataset_split_idxs = []

    dataset_idxs = data_cfg.datasets + ["all"] if len(data_cfg.datasets) > 1 else data_cfg.datasets
    for dataset_idx in dataset_idxs:
        if data_cfg.fold in list(range(10)):
            dataset_split_idxs.append(dataset_idx + "_" + "val")
        elif data_cfg.fold in ["all", "sanity"]:
            dataset_split_idxs.append(dataset_idx + "_" + data_cfg.fold)
        else:
            if "val" in data_cfg.split[data_cfg.datasets[0]]:
                dataset_split_idxs.append(dataset_idx + "_" + "val")
            dataset_split_idxs.append(dataset_idx + "_" + "test")

    return dataset_split_idxs


def get_loss_weights_and_flags(data_cfg, run_cfg, dataset_train=None):
    """ get class and confusion weights for training """
    class_weight = np.ones(len(data_cfg.scored_classes), dtype=np.float32)
    if run_cfg.class_weight and dataset_train is not None:
        Y = np.stack(dataset_train.Y, 0)
        class_weight = np.sum(Y, axis=0).astype(np.float32)
        class_weight[class_weight == 0] = 1
        max_num = np.max(class_weight)
        for i in range(len(class_weight)):
            class_weight[i] = np.sqrt(max_num / class_weight[i])

    confusion_weight_flag = run_cfg.confusion_weight
    confusion_weight = load_weights(data_cfg.path + "weights.csv", data_cfg.scored_classes)

    return class_weight, confusion_weight, confusion_weight_flag


def collate_into_list(args):
    """ collate variable-length ecg signals into list """
    X = [a[0] for a in args]
    Y = torch.stack([a[1] for a in args], 0)
    return X, Y


def collate_into_block(batch, l, stride):
    """
    collate variable-length ecg signals into block
    for those longer than chunk_length, divide them into l-point chunks with (overlapping) stride
    """
    X, Y = batch
    if stride is None:
        # assume all ecg signals have same length
        X_block = torch.stack(X, 0)
        X_flag  = X[0].new_ones((len(X)), dtype=torch.bool)
    else:
        # collate variable-length ecg signals
        b, c, = 0, X[0].shape[0]
        for x in X:
            b += int(np.ceil((x.shape[1] - l) / float(stride) + 1))

        X_block = X[0].new_zeros((b, c, l))
        X_flag  = X[0].new_zeros((b), dtype=torch.bool)
        idx = 0
        for x in X:
            num_chunks = int(np.ceil((x.shape[1] - l) / float(stride) + 1))
            for i in range(num_chunks):
                if i != num_chunks - 1:
                    X_block[idx] = x[:, i*stride:i*stride + l]
                    X_flag[idx] = False
                elif x.shape[1] > l:
                    X_block[idx] = x[:, -l:]
                    X_flag[idx] = True
                else:
                    X_block[idx, :, :x.shape[1]] = x[:, :]
                    X_flag[idx] = True
                idx += 1

    return X_block, X_flag, Y
