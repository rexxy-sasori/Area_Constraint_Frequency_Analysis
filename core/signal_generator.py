import numpy as np

from core import noise_generator
from core import utils as dsputils


class InputSignalGenerator:
    def __init__(self, signal_configs, noise_configs):
        """
        :param signal_configs: configuration for signal content
        :param noise_configs: configuration for noise content
        """
        self.fs = signal_configs.fs
        self.Ts = 1 / self.fs

        self.phases = signal_configs.phases
        self.freqs = signal_configs.freqs
        self.amps = signal_configs.amps

        self.num_pos_sample = signal_configs.num_pos_decision * signal_configs.block_size * signal_configs.num_blocks_avg
        self.observation_block_size = signal_configs.block_size * signal_configs.num_blocks_avg
        self.N = signal_configs.block_size
        self.L = signal_configs.num_blocks_avg
        self.hop_size = signal_configs.hop_size

        self.t = np.arange(self.num_pos_sample) * self.Ts

        noise_gen_cls = noise_generator.NOISE_CLS.get(noise_configs.name)
        if noise_gen_cls is None:
            raise NotImplementedError('noise type not recognized: {}'.format(noise_configs.name))

        self.noise_generator = noise_gen_cls(self.fs / 2, self.observation_block_size, noise_configs)
        self.input_snr = 0
        self.noise_cov = None

    def get(self):
        """
        get input signal = rider + noise
        :return:
        """
        rider = self.__get_rider_signal__()
        nonrider = np.zeros_like(rider)
        rider_with_null = np.concatenate((rider, nonrider), axis=0)
        noise = self.__get_noise__(rider_with_null.shape)
        ret = rider_with_null + noise
        labels = np.concatenate((np.ones(rider.shape[0]), -np.ones(nonrider.shape[0])))

        ret = [(vec_sample, label) for vec_sample, label in zip(ret, labels)]
        # np.random.shuffle(ret) # shuffle the dataset
        observations_time = np.array([observation for observation, _ in ret])
        labels = np.array([label for _, label in ret])

        # rider_fft = np.fft.fft(rider,axis=1)
        # noise_fft = np.fft.fft(self.__get_noise__(rider.shape), axis=1)
        # self.input_snr = dsputils.snr(np.sum((np.abs(rider_fft)**2).mean(0)), np.sum((np.abs(noise_fft)**2).mean(0)))
        observations_time_reshape = observations_time.reshape(len(observations_time), self.L, self.N)
        return observations_time_reshape, labels

    def __get_rider_signal__(self):
        """
        :return: rider signal = \sum_{i=0}^{i=N-1}{amp[i] * cos(2\pi \times freqs[i] * t + phases[i]}
        """
        ret = 0
        for amp, freq, phase in zip(self.amps, self.freqs, self.phases):
            ret += amp * np.cos(2 * np.pi * freq * self.t + phase)

        ret = dsputils.reformat(ret, self.observation_block_size, self.hop_size, self.num_pos_sample)
        return ret

    def __get_noise__(self, signal_shape):
        """
        get noise with the same shape as the signal
        :param signal_shape:
        :return:
        """
        noise = self.noise_generator.get(signal_shape)
        self.noise_cov = np.cov(noise.T)
        return noise
