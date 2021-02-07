import math

import numpy as np
from scipy.special import comb
from sklearn.metrics import roc_curve

from IO import result
from core import utils


class HarmonicEstimator:
    def __init__(self, detection_config, input_signal_generator):
        self.input_signal_gnerator = input_signal_generator
        self.rule = detection_config.name

    def get_roc(self, observations, labels):
        if self.rule == 'ml':
            observations = observations.mean(1)
            dim = observations.shape[-1]
            observations_half = observations[:, 0:int(dim / 2)]
            scores = np.max(observations_half, axis=1)
            indices = np.argmax(observations_half, axis=1)
            mlestimates = MLEstimates(scores, indices)
        elif self.rule == 'switched':
            dim = observations.shape[-1]
            observations = observations[:, :, 0:int(dim / 2)]
            scores = observations.sum(2).mean(1)
            Nd = len(observations)
            indices, _ = utils.mode(np.argmax(observations, axis=2), axis=1)
            estimation_scores = np.array([np.mean(observations[i, :, indices[i]]) for i in range(Nd)])
            mlestimates = MLEstimates(estimation_scores, indices)
        elif self.rule == 'lmpi':
            scores = (observations ** 2).sum(1)  # lmpi
        elif self.rule == 'umpi':  # umpi
            snr = 1 / self.input_signal_gnerator.noise_generator.steady_state
            scores = np.zeros(observations.shape[0])
            for idx in range(observations.shape[1]):
                scores += self.compute_umpi_stats(observations, idx, snr)
        elif self.rule == 'single':
            observations = observations.mean(1)
            dim = observations.shape[-1]
            observations_half = observations[:, 0:int(dim / 2)]
            scores = observations_half[:, 6] #Todo extract to parameter level
            indices = np.argmax(observations_half, axis=1)
            mlestimates = MLEstimates(scores, indices)
        else:
            raise NotImplementedError

        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc = result.ROC(fpr, tpr, thresholds)
        return roc, mlestimates

    def compute_umpi_stats(self, observations, idx, snr):
        curr = np.sum(np.exp(snr * observations) * (snr * observations) ** idx, axis=1)
        weight = comb(16, idx, exact=True) / (math.factorial(idx))
        return weight * curr


class MLEstimates:
    def __init__(self, scores, indices):
        self.scores = scores
        self.indices = indices
