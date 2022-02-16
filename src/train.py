# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# Gradual warmup scheduler (https://github.com/ildoonet/pytorch-gradual-warmup-lr)

import os
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from src.data import collate_into_block
from src.utils import Print
import src.evaluate as evaluate


class Trainer():
    """ train / eval helper class """
    def __init__(self, model, data_cfg, multilabel_flag, loss_weights_and_flags):
        self.model = model
        self.chunk_length = data_cfg.chunk_length
        self.chunk_stride = data_cfg.chunk_stride
        self.scored_classes = data_cfg.scored_classes
        self.normal_class = data_cfg.normal_class
        self.multilabel_flag = multilabel_flag
        self.class_weight, self.confusion_weight, self.confusion_weight_flag = loss_weights_and_flags
        self.optim = None
        self.scheduler = None
        self.mixup_flag = None
        self.device = None

        # initialize logging parameters
        self.train_flag = False
        self.val_flag = False if data_cfg.fold == "all" else True
        self.epoch = 0.0
        self.best_loss = None
        self.logger_train = Logger()
        self.logger_eval  = Logger()

    def train(self, batch):
        # training of the model
        if self.mixup_flag == "manifold":  batch, lam, p = self.mixup(batch, manifold=True)
        elif self.mixup_flag == "input":   batch, lam, p = self.mixup(batch, manifold=False)
        else:                              batch, lam, p = batch, None, None
        batch = collate_into_block(batch, self.chunk_length, self.chunk_stride)
        batch = set_device(batch, self.device)

        self.model.train()
        inputs, flags, labels = batch

        self.optim.zero_grad()
        if self.mixup_flag == "manifold": outputs = self.model(inputs, flags, lam, p)
        else:                             outputs = self.model(inputs, flags)
        loss = get_loss(outputs, labels, self.class_weight, self.confusion_weight,
                        self.multilabel_flag, self.confusion_weight_flag)
        loss.backward()
        self.optim.step()

        # logging
        self.logger_train.update(len(inputs), loss.item())

    def evaluate(self, batch):
        # evaluation of the model
        batch = collate_into_block(batch, self.chunk_length, self.chunk_stride)
        batch = set_device(batch, self.device)

        self.model.eval()
        inputs, flags, labels = batch

        with torch.no_grad():
            outputs = self.model(inputs, flags)
            loss = get_loss(outputs, labels, self.class_weight, self.confusion_weight,
                            self.multilabel_flag, self.confusion_weight_flag)
        # logging
        if self.multilabel_flag:
            scalar_outputs = torch.sigmoid(outputs)
            binary_outputs = scalar_outputs > 0.5
        else:
            scalar_outputs = torch.softmax(outputs, 1)
            max_idx = torch.argmax(scalar_outputs, dim=1, keepdim=True)
            binary_outputs = torch.zeros_like(scalar_outputs, dtype=torch.bool).scatter_(1, max_idx, True)

        self.logger_eval.update(len(inputs), loss.item())
        self.logger_eval.keep(labels, scalar_outputs, binary_outputs)

    def mixup(self, batch, manifold=True):
        # mixup training
        inputs, labels = batch

        lengths = np.array([x.shape[1] for x in inputs])
        lam = np.random.beta(0.2, 0.2)
        p = np.random.permutation(len(lengths))

        if manifold:
            inputs_m = inputs
        else:
            lengths_m = np.maximum(lengths, lengths[p])
            inputs_m = []
            for i in range(len(lengths_m)):
                l, l0, l1 = lengths_m[i], lengths[i], lengths[p[i]]
                offset0 = np.random.randint(l - l0 + 1)
                offset1 = np.random.randint(l - l1 + 1)
                input = inputs[i].new_zeros(inputs[i].shape[0], l)
                input[:, offset0:offset0+l0] += lam * inputs[i]
                input[:, offset1:offset1+l1] += (1 - lam) * inputs[p[i]]
                inputs_m.append(input)

        labels = lam * labels + (1 - lam) * labels[p]

        return (inputs_m, labels), lam, p

    def scheduler_step(self):
        # scheduler_step
        self.scheduler.step(self.logger_eval.get_loss())

    def save_model(self, save_prefix):
        # save state_dicts to checkpoint """
        if save_prefix is None: return
        elif not os.path.exists(save_prefix + "/checkpoints/"):
            os.makedirs(save_prefix + "/checkpoints/", exist_ok=True)

        state = {}
        state["model"] = self.model.state_dict()
        state["optim"] = self.optim.state_dict()
        state["scheduler"] = self.scheduler.state_dict()
        state["epoch"] = self.epoch
        torch.save(state, save_prefix + "/checkpoints/%d.pt" % self.epoch)

        loss = self.logger_eval.get_loss()
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            torch.save(state, save_prefix + "/checkpoints/best_loss.pt")

    def load_model(self, checkpoint, save_prefix, output):
        # load state_dicts from checkpoint """
        if checkpoint is None:
            if save_prefix is None or not os.path.exists(save_prefix + "/checkpoints/"): return
            checkpoints = [os.path.splitext(file)[0] for file in os.listdir(save_prefix + "/checkpoints/")]
            checkpoints = sorted([int(checkpoint) for checkpoint in checkpoints if not checkpoint.startswith("best")])
            if len(checkpoints) == 0: return
            checkpoint = save_prefix + "/checkpoints/%d.pt" % checkpoints[-1]
            Print('resuming from the last checkpoint [%s]' % (checkpoint), output)

        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        state_dict = OrderedDict()
        for k, v in checkpoint["model"].items():
            if k.startswith("module."): k = k[7:]
            state_dict[k] = v
        self.model.load_state_dict(state_dict, strict=False)
        if self.optim is not None and "optim" in checkpoint:
            Print('loading a optim state_dict from the checkpoint', output)
            self.optim.load_state_dict(checkpoint["optim"])
            Print('loading a scheduler state_dict from the checkpoint', output)
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            Print('loading current epoch from the checkpoint', output)
            self.epoch = checkpoint["epoch"]

    def save_outputs(self, idx, save_prefix):
        # save validation output
        if save_prefix is None: return
        if not os.path.exists(save_prefix + "/outputs/"):
            os.makedirs(save_prefix + "/outputs/", exist_ok=True)
        self.logger_eval.aggregate()
        np.save(save_prefix + "/outputs/%s_labels.npy" % (idx), self.logger_eval.labels)
        np.save(save_prefix + "/outputs/%s_scalar_outputs.npy" % (idx), self.logger_eval.scalar_outputs)
        np.save(save_prefix + "/outputs/%s_binary_outputs.npy" % (idx), self.logger_eval.binary_outputs)

    def set_device(self, device, data_parallel):
        # set gpu configurations
        self.device = device
        self.model = self.model.to(self.device)
        if data_parallel:
            self.model = nn.DataParallel(self.model)

        if self.class_weight is not None:
            self.class_weight = torch.from_numpy(self.class_weight).to(self.device)
        if self.confusion_weight is not None:
            self.confusion_weight = torch.from_numpy(self.confusion_weight).to(self.device)

    def set_optim_scheduler(self, run_cfg, params):
        # set optim and scheduler for training
        optim, scheduler = get_optim_scheduler(run_cfg, params)
        self.train_flag = True
        self.optim = optim
        self.scheduler = scheduler
        self.mixup_flag = run_cfg.mixup

    def get_headline(self):
        # get a headline for logging
        headline = []
        if self.train_flag:
            headline += ["ep", "split"]
            headline += self.logger_train.get_headline(loss_only=True)
            if self.val_flag: headline += ["|"]

        if self.val_flag:
            headline += ["split"]
            headline += self.logger_eval.get_headline(loss_only=False)

        return "\t".join(headline)

    def log(self, idx, output, writer):
        # logging
        log, log_dict = [], {}

        if self.train_flag:
            self.logger_train.evaluate()
            log += ["%03d" % self.epoch, "train"]
            log += self.logger_train.log
            if writer is not None:
                for k, v in self.logger_train.log_dict.items():
                    if k not in log_dict: log_dict[k] = {}
                    log_dict[k]["train"] = v
            if self.val_flag: log += ["|"]

        if self.val_flag:
            self.logger_eval.evaluate(self.scored_classes, self.normal_class, self.confusion_weight)
            log += [idx]
            log += self.logger_eval.log
            if writer is not None:
                for k, v in self.logger_eval.log_dict.items():
                    if k not in log_dict: log_dict[k] = {}
                    log_dict[k][idx] = v

        Print("\t".join(log), output)
        if writer is not None:
            for k, v in log_dict.items():
                writer.add_scalars(k, v, self.epoch)
            writer.flush()

        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_eval.reset()


class Logger():
    """ Logger class """
    def __init__(self):
        self.total = 0.0
        self.loss = 0.0
        self.labels = []
        self.scalar_outputs = []
        self.binary_outputs = []
        self.log = []
        self.log_dict = {}

    def update(self, total, loss):
        # update logger for current mini-batch
        self.total += total
        self.loss += loss * total

    def keep(self, labels, scalar_outputs, binary_outputs):
        # keep labels and outputs for future computations
        self.labels.append(labels.cpu().detach().numpy())
        self.scalar_outputs.append(scalar_outputs.cpu().detach().numpy())
        self.binary_outputs.append(binary_outputs.cpu().detach().numpy())

    def get_loss(self):
        # get current averaged loss
        loss = self.loss / self.total if self.total > 0 else 0
        return loss

    def get_headline(self, loss_only):
        # get headline
        headline = ["loss"]
        if not loss_only: headline += ["acc", "f1", "f2", "g2", "auroc", "aupr", "pnc"]

        return headline

    def evaluate(self, scored_classes=None, normal_class=None, confusion_weight=None):
        # compute evaluation metrics
        self.aggregate()
        metrics = ["loss"]
        evaluations = [self.get_loss()]

        if scored_classes is not None:
            metrics += ["acc", "f1", "f2", "g2", "auroc", "aupr", "pnc"]
            evaluations += [
                evaluate.compute_accuracy(self.labels, self.binary_outputs),
                evaluate.compute_f_measure(self.labels, self.binary_outputs),
                *evaluate.compute_beta_measures(self.labels, self.binary_outputs, beta=2),
                *evaluate.compute_auc(self.labels, self.scalar_outputs),
                evaluate.compute_challenge_metric(confusion_weight.cpu().numpy(), self.labels, self.binary_outputs,
                                                  scored_classes, normal_class)
            ]
        self.log = ["%.4f" % eval for eval in evaluations]
        self.log_dict = {metric:eval for metric, eval in zip(metrics, evaluations)}

    def aggregate(self):
        # aggregate kept labels and outputs
        if isinstance(self.labels, list) and len(self.labels) > 0:
            self.labels = np.concatenate(self.labels, axis=0)
        if isinstance(self.scalar_outputs, list) and len(self.scalar_outputs) > 0:
            self.scalar_outputs = np.concatenate(self.scalar_outputs, axis=0)
        if isinstance(self.binary_outputs, list) and len(self.binary_outputs) > 0:
            self.binary_outputs = np.concatenate(self.binary_outputs, axis=0)

    def reset(self):
        # reset logger
        self.total = 0.0
        self.loss = 0.0
        self.labels = []
        self.scalar_outputs = []
        self.binary_outputs = []
        self.log = []
        self.log_dict = {}


def get_optim_scheduler(cfg, params):
    """ configure optim and scheduler """
    if cfg.optimizer == "adam":
        optim = torch.optim.Adam([{'params': params[0], 'weight_decay': cfg.weight_decay},
                                  {'params': params[1], 'weight_decay': 0}],
                                 lr=cfg.learning_rate)
    else:
        optim = torch.optim.SGD([{'params': params[0], 'weight_decay': cfg.weight_decay},
                                 {'params': params[1], 'weight_decay': 0}],
                                lr=cfg.learning_rate, momentum=cfg.momentum, nesterov=True)

    if cfg.scheduler.startswith("warmup"):
        warmup_epochs = int(cfg.num_epochs * 0.1)
        after_epochs = int(cfg.num_epochs * 0.9)
    else:
        warmup_epochs = 0
        after_epochs = cfg.num_epochs

    if cfg.scheduler.endswith("plateau"):
        after_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, "min", cfg.learning_rate_decay, cfg.patience)
        value_flag = True
    elif cfg.scheduler.endswith("cosine"):
        after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, after_epochs)
        value_flag = False
    else:
        after_scheduler = None
        value_flag = False

    scheduler = GradualWarmupScheduler(optim, warmup_epochs, after_scheduler, value_flag)

    return optim, scheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epoch: target learning rate is reached at warmup_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
        value_flag: whether value is required for after_scheduler step
    """

    def __init__(self, optimizer, warmup_epoch, after_scheduler, value_flag):
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = after_scheduler
        self.value_flag = value_flag
        self.finished = False if warmup_epoch != 0 else True
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch and self.warmup_epoch != 0:
            if self.last_epoch == self.warmup_epoch - 1:
                self.finished = True
            return [base_lr * ((float(self.last_epoch) + 1) / self.warmup_epoch) for base_lr in self.base_lrs]
        elif self.after_scheduler is not None:
            return [group['lr'] for group in self.after_scheduler.optimizer.param_groups]
        else:
            return self.base_lrs

    def step(self, value=None):
        if not self.finished:
            super(GradualWarmupScheduler, self).step()
        else:
            if self.value_flag: self.after_scheduler.step(value)
            else:               self.after_scheduler.step()


def get_loss(outputs, labels, class_weight, confusion_weight, multilabel_flag, confusion_weight_flag):
    """ get (binary) cross entropy loss with/without class_weight & confusion_weight """
    if multilabel_flag and confusion_weight_flag:
        # get confusion_weight for current labels
        conf_weight = torch.sum(labels.unsqueeze(2) * (1 - confusion_weight).unsqueeze(0), 1)
        num_labels = torch.sum(labels, 1, keepdim=True)

        # normalization in terms of num_labels (set conf weight 0.5 for all-negative labeled samples)
        conf_weight[num_labels.squeeze() == 0] = 0.5
        num_labels[num_labels == 0] = 1
        conf_weight = conf_weight / num_labels

        loss = -torch.mean(class_weight * labels * F.logsigmoid(outputs) +
                           conf_weight * (1 - labels) * F.logsigmoid(-outputs))
    elif multilabel_flag:
        loss = -torch.mean(class_weight * labels * F.logsigmoid(outputs) + (1 - labels) * F.logsigmoid(-outputs))
    else:
        loss = -torch.mean(torch.sum(class_weight * labels * F.log_softmax(outputs, dim=1)), dim=0)

    return loss


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
