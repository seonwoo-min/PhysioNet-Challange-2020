#!/usr/bin/env python

import os
import numpy as np
from collections import OrderedDict

import torch

import src.config as config
from src.data import get_dataset_from_configs, collate_into_list, get_loss_weights_and_flags
from src.model.model_utils import get_model
from src.train import Trainer


def load_12ECG_model(output_training_directory):
    # load the model from disk
    data_cfg = config.DataConfig("config/data.json")
    preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
    model_cfg = config.ModelConfig("config/model.json")
    run_cfg = config.RunConfig("config/run.json")

    models, thresholds = [], []
    for fold in range(10):
        model, _ = get_model(model_cfg, data_cfg.num_channels, len(data_cfg.scored_classes))
        checkpoint = torch.load(os.path.join(output_training_directory, 'finalized_model_%d.sav' % fold),
                                map_location=torch.device("cpu"))
        state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
        models.append(model)

        threshold = np.load(os.path.join(output_training_directory, 'finalized_model_thresholds_%d.npy' % fold))
        thresholds.append(threshold)

    thresholds = np.average(np.stack(thresholds, axis=0), axis=0)
    eval_list = [data_cfg, preprocess_cfg, run_cfg, models, thresholds]

    return eval_list


def run_12ECG_classifier(data, header, eval_list):
    # Use your classifier here to obtain a label and score for each class.
    data_cfg, preprocess_cfg, run_cfg, models, thresholds = eval_list
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg.data = data
    data_cfg.header = header
    dataset_val = get_dataset_from_configs(data_cfg, preprocess_cfg)
    iterator_val = torch.utils.data.DataLoader(dataset_val, 1, collate_fn=collate_into_list)

    outputs_list = []
    for model in models:
        loss_weights_and_flags = get_loss_weights_and_flags(data_cfg, run_cfg)
        trainer = Trainer(model, data_cfg, run_cfg.multilabel, loss_weights_and_flags)
        trainer.set_device(device, data_parallel=False)

        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch)

        outputs = trainer.logger_eval.scalar_outputs[0][0]
        outputs_list.append(outputs)

    classes = data_cfg.scored_classes
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    for i in range(num_classes):
        for j in range(len(outputs_list)):
            current_score[i] += outputs_list[j][i]
        current_score[i] = current_score[i] / len(outputs_list)
        if current_score[i] > thresholds[i]: current_label[i] = 1

    return current_label, current_score, classes

