import os
from datetime import datetime

import torch


class DetectionResult:
    def __init__(self, roc=None, usr_configs=None, input_signal_sq_mag=None, input_snr=None, estimator=None,
                 scores=None):
        self.usr_configs = usr_configs
        self.roc = roc
        self.input_signal_sq_mag = input_signal_sq_mag
        self.input_snr = input_snr
        self.estimator = estimator
        self.scores = scores

    def save(self, dir_name, remote=False):
        if remote:
            self.save_remote(dir_name)
        else:
            self.save_local(dir_name)

    def save_remote(self, dir_name):
        pass

    def save_local(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        now = datetime.now()
        torch.save({'result': self}, os.path.join(dir_name, 'result_{}.pt.tar'.format(now)))


class ROC:
    def __init__(self, fpr, tpr, thresholds):
        self.fpr = fpr
        self.tpr = tpr
        self.thresholds = thresholds
