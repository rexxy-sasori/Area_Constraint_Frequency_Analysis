import numpy as np

from core import noise_generator
from core import utils as dsputils
from core import freq_transform
from IO.config import UsrConfigs

pi = np.pi


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

        # normalize the signal such that each block has power of 1
        # if ret.std() != 0:
        ret = ret / ret.std()
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


def round_idx(float):
    floor = np.floor(float)
    ceil = np.ceil(float)

    if float - floor >= ceil - float:
        return int(ceil)
    else:
        return int(floor)


def get_output_signal(L, N, freq_o, fs, phi, kernel='fft', transform=False):
    signal_configs = UsrConfigs({})
    noise_configs = UsrConfigs({})
    transform_configs = UsrConfigs({})
    init_args = UsrConfigs({})
    Nd = 10000

    setattr(signal_configs, 'fs', fs)
    setattr(signal_configs, 'num_pos_decision', Nd)
    setattr(signal_configs, 'block_size', N)
    setattr(signal_configs, 'num_blocks_avg', L)
    setattr(signal_configs, 'hop_size', N * L)
    setattr(signal_configs, 'freqs', [freq_o])
    setattr(signal_configs, 'phases', [phi])
    setattr(signal_configs, 'amps', [1])
    setattr(noise_configs, 'name', 'rvs')
    setattr(init_args, 'slope', 1)
    setattr(init_args, 'steady_state', 0)
    setattr(init_args, 'top', 0)
    setattr(noise_configs, 'init_args', init_args)
    setattr(transform_configs, 'name', kernel)
    generator = InputSignalGenerator(signal_configs, noise_configs)
    input_signal, _ = generator.get()
    signal = input_signal[0:Nd, :, :]
    observations, _ = freq_transform.transform_all(signal, transform_configs, signal_configs) if transform else None

    return input_signal, observations


def get_output_composite(freq_o, phase, L, N, noise_level, kernel='fft', transform=False):
    signal_configs = UsrConfigs({})
    noise_configs = UsrConfigs({})
    transform_configs = UsrConfigs({})
    init_args = UsrConfigs({})
    Nd = 10000

    setattr(signal_configs, 'fs', 2000)
    setattr(signal_configs, 'num_pos_decision', Nd)
    setattr(signal_configs, 'block_size', N)
    setattr(signal_configs, 'num_blocks_avg', L)
    setattr(signal_configs, 'hop_size', N * L)
    setattr(signal_configs, 'freqs', [freq_o])
    setattr(signal_configs, 'phases', [phase])
    setattr(signal_configs, 'amps', [1])
    setattr(noise_configs, 'name', 'rvs')
    setattr(init_args, 'slope', 1)
    setattr(init_args, 'steady_state', noise_level)
    setattr(init_args, 'top', noise_level)
    setattr(noise_configs, 'init_args', init_args)
    setattr(transform_configs, 'name', kernel)
    generator = InputSignalGenerator(signal_configs, noise_configs)
    input_signal, labels = generator.get()
    noise = input_signal
    observations, _ = freq_transform.transform_all(noise, transform_configs, signal_configs) if transform else None

    return input_signal, observations, labels


def get_output_signal_power(freq_o, phi, kernel, fs=2000, N=16, L=1):
    _, sm_signal = get_output_signal(L, N, freq_o, fs, phi, kernel, True)
    bin_idx = round_idx(freq_o * N / fs)

    output_signal_power = sm_signal[:, :, bin_idx].mean()

    return output_signal_power


def get_output_noise_power(freq_o, noise_level, kernel, fs=2000, N=16, L=1):
    _, sm_noise = get_output_noise(L, N, noise_level, kernel, True)
    bin_idx = round_idx(freq_o * N / fs)
    output_noise_power = sm_noise[:, :, bin_idx].mean()

    return output_noise_power
