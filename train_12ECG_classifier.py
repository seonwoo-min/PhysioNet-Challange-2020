#!/usr/bin/env python

import sys
import os
import multiprocessing
import numpy as np
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch

import src.config as config
from src.data import get_dataset_from_configs, collate_into_list, get_loss_weights_and_flags
from src.model.model_utils import get_model, get_profile
from src.train import Trainer
from src.evaluate import find_best_threshold
from src.utils import set_seeds


def get_filenames_from_split(data_cfg, data_fold, split_idx, input_training_directory):
    filenames_all = {}
    for file in os.listdir(input_training_directory):
        path = input_training_directory
        if os.path.isfile(os.path.join(path, file)):
            if not file.lower().startswith('.') and file.lower().endswith('mat'):
                filename = os.path.splitext(file)[0]
                filenames_all[filename] = path
        else:
            path = os.path.join(path, file)
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path, file)):
                    if not file.lower().startswith('.') and file.lower().endswith('mat'):
                        filename = os.path.splitext(file)[0]
                        filenames_all[filename] = path

    filenames_split = []
    for dataset in data_cfg.datasets:
        if split_idx == "train":
            for fold in range(10):
                if fold != data_fold:
                    filenames_split += data_cfg.split[dataset]["fold%d" % fold]
        else:
            filenames_split += data_cfg.split[dataset]["fold%d" % data_fold]

    filenames = []
    for filename in filenames_split:
        if filename in filenames_all:
            path = filenames_all[filename]
            filenames.append(os.path.join(path, filename))

    return filenames


def train_12ECG_classifier(input_training_directory, output_training_directory):
    if not os.path.isdir(output_training_directory):
        os.mkdir(output_training_directory)

    data_cfg = config.DataConfig("config/data.json")
    preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
    model_cfg = config.ModelConfig("config/model.json")
    run_cfg = config.RunConfig("config/run.json", eval=False)
    #os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Training
    for fold in range(10):
        print('Loading data %d...' % fold)
        set_seeds(2020)
        data_cfg.filenames = get_filenames_from_split(data_cfg, fold, "train", input_training_directory)
        dataset_train = get_dataset_from_configs(data_cfg, preprocess_cfg, split_idx="train")
        data_cfg.filenames = get_filenames_from_split(data_cfg, fold, "val", input_training_directory)
        dataset_val = get_dataset_from_configs(data_cfg, preprocess_cfg, split_idx="val")
        iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size_train, collate_fn=collate_into_list,
                                                     shuffle=True, pin_memory=True, num_workers=4)
        iterator_val = torch.utils.data.DataLoader(dataset_val, run_cfg.batch_size_eval, collate_fn=collate_into_list,
                                                   shuffle=False, pin_memory=True, num_workers=4)
        print("training samples: %d" % len(dataset_train))
        print("evaluation samples: %d" % len(dataset_val))

        ## initialize a model
        model, params = get_model(model_cfg, data_cfg.num_channels, len(data_cfg.scored_classes))
        get_profile(model, data_cfg.num_channels, data_cfg.chunk_length)

        ## setup trainer configurations
        loss_weights_and_flags = get_loss_weights_and_flags(data_cfg, run_cfg, dataset_train)
        trainer = Trainer(model, data_cfg, run_cfg.multilabel, loss_weights_and_flags)
        trainer.set_device(device, data_parallel=False)
        trainer.set_optim_scheduler(run_cfg, params)

        ## train a model
        print('Training model %d...' % fold)
        PNC_list = []
        for epoch in range(int(trainer.epoch), run_cfg.num_epochs):
            for B, batch in enumerate(iterator_train):
                trainer.train(batch)
                if B % 5 == 0: print('# epoch [{}/{}] train {:.1%}'.format(
                    epoch + 1, run_cfg.num_epochs, B / len(iterator_train)), end='\r', file=sys.stderr)
            print(' ' * 150, end='\r', file=sys.stderr)

            for B, batch in enumerate(iterator_val):
                trainer.evaluate(batch)
                if B % 5 == 0: print('# epoch [{}/{}] val {:.1%}'.format(
                    epoch + 1, run_cfg.num_epochs, B / len(iterator_val)), end='\r', file=sys.stderr)
            print(' ' * 150, end='\r', file=sys.stderr)

            ### print log and save models
            trainer.epoch += 1
            trainer.scheduler_step()
            torch.save(trainer.model.state_dict(), os.path.join(output_training_directory, '%d_epoch%d.sav' % (fold, epoch)))
            trainer.logger_eval.evaluate(trainer.scored_classes, trainer.normal_class, trainer.confusion_weight)
            PNC_list.append(float(trainer.logger_eval.log[-1]))
            print("Epoch: %03d  PNC: %s" % (epoch, trainer.logger_eval.log[-1]))
            trainer.log_reset()

        ## save a model
        print('Saving model %d...' % fold)
        fold_model = os.path.join(output_training_directory, '%d_*' % (fold))
        best_model = os.path.join(output_training_directory, '%d_epoch%d.sav' % (fold, np.argmax(PNC_list)))
        final_model = os.path.join(output_training_directory, 'finalized_model_%d.sav' % fold)
        os.system("cp %s %s" % (best_model, final_model))
        os.system("rm -rf %s" % fold_model)

        ## find best class-specific threholds
        trainer.model.load_state_dict(torch.load(final_model, map_location=device), strict=False)
        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch)
            if B % 5 == 0: print('# val {:.1%}'.format(B / len(iterator_val)), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)
        trainer.logger_eval.aggregate()

        arguments = [(trainer.confusion_weight.cpu().numpy(),
                      trainer.logger_eval.labels, trainer.logger_eval.scalar_outputs,
                      trainer.scored_classes, trainer.normal_class, i) for i in range(len(trainer.scored_classes))]
        thresholds = np.array(multiprocessing.Pool(4).starmap(find_best_threshold, arguments))
        final_model_thresholds = os.path.join(output_training_directory, 'finalized_model_thresholds_%d.npy' % fold)
        np.save(final_model_thresholds, thresholds)