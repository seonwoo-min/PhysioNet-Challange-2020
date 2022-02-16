# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
import scipy.signal as sig
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


def preprocess_signal(x, preprocess_cfg, sample_rate):
    """ resample, filter, scale, ecg signal """
    if sample_rate != preprocess_cfg.sample_rate:
        num = x.shape[1] // sample_rate * preprocess_cfg.sample_rate
        x = sig.resample(x, num, axis=1)
        sample_rate = preprocess_cfg.sample_rate
    x = filter_signal(x, preprocess_cfg, sample_rate)
    x = scale_signal(x, preprocess_cfg)

    return x


def filter_signal(x, preprocess_cfg, sample_rate):
    """ filter ecg signal """
    nyq = sample_rate * 0.5
    for i in range(len(x)):
        for cutoff in preprocess_cfg.filter_highpass:
            x[i] = sig.filtfilt(*sig.butter(2, cutoff / nyq, btype='highpass'), x[i])
        for cutoff in preprocess_cfg.filter_lowpass:
            if cutoff >= nyq: cutoff = nyq - 0.05
            x[i] = sig.filtfilt(*sig.butter(2, cutoff / nyq, btype='lowpass'), x[i])
        for cutoff in preprocess_cfg.filter_bandpass:
            x[i] = sig.filtfilt(*sig.butter(2, [cutoff[0] / nyq, cutoff[1] / nyq], btype='bandpass'), x[i])
        for cutoff in preprocess_cfg.filter_notch:
            x[i] = sig.filtfilt(*sig.iirnotch(cutoff, cutoff, sample_rate), x[i])

    return x


def scale_signal(x, preprocess_cfg):
    """ scale ecg signal """
    for i in range(len(x)):
        if preprocess_cfg.scaler is None: continue
        elif "minmax" in preprocess_cfg.scaler:   scaler = MinMaxScaler()
        elif "standard" in preprocess_cfg.scaler: scaler = StandardScaler()
        elif "robust" in preprocess_cfg.scaler:   scaler = RobustScaler()
        scaler.fit(np.expand_dims(x[i], 1))
        x[i] = scaler.transform(np.expand_dims(x[i], 1)).squeeze()

    return x


def augment_signal(x):
    """ augmentations (scale, noise) """
    for i in range(len(x)):
        scale = np.random.normal(loc=1.0, scale=0.1)
        noise = np.random.normal(loc=0.0, scale=0.1, size=x[i].shape)
        x[i] = x[i] * scale + noise

    return x


def preprocess_label(labels, scored_classes, equivalent_classes):
    """ convert string labels to binary labels """
    y = np.zeros((len(scored_classes)), np.float32)
    for label in labels:
        if label in equivalent_classes:
            label = equivalent_classes[label]

        if label in scored_classes:
            y[scored_classes.index(label)] = 1

    return y