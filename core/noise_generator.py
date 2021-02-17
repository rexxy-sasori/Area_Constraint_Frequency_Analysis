import numpy as np


class NoiseGenerator:
    def __init__(self, half_fs, block_size, noise_configs):
        self.check_args(noise_configs)
        self.block_size = block_size
        self.half_fs = half_fs

    def check_args(self, noise_configs):
        return NotImplementedError

    def get(self, shape):
        return NotImplementedError


class WhiteNoiseGenerator(NoiseGenerator):
    def __init__(self, half_fs, block_size, noise_configs):
        super(WhiteNoiseGenerator, self).__init__(half_fs, block_size, noise_configs)
        self.noise_power = noise_configs.init_args.noise_power

    def check_args(self, noise_configs):
        pass

    def get(self, signal_shape):
        ret = np.random.normal(0, np.sqrt(self.noise_power / self.block_size), signal_shape)
        return ret


class RVSNoiseGenerator(NoiseGenerator):
    def __init__(self, half_fs, block_size, noise_configs):
        super(RVSNoiseGenerator, self).__init__(half_fs, block_size, noise_configs)
        self.steady_state = noise_configs.init_args.steady_state
        self.slope = noise_configs.init_args.slope
        self.top = noise_configs.init_args.top

        f = np.linspace(0, self.half_fs, int(self.block_size / 2) + 1)
        f_below1 = f[f < 1]
        f_above1 = f[f >= 1]
        f_below1_response = self.top * np.ones_like(f_below1)
        f_above1_response = self.steady_state + (self.top - self.steady_state) * np.power(f_above1, -self.slope)
        response_half = np.concatenate((f_below1_response, f_above1_response))
        self.response = np.concatenate((response_half, response_half[1:-1:][::-1]))
        self.f = f

    def check_args(self, noise_configs):
        # assert noise_configs.init_args.steady_state <= 10 and noise_configs.init_args.slope > 0
        return

    def get(self, signal_shape):
        uncorrelated_noise = np.random.normal(0, 1, signal_shape)
        ret = np.zeros(signal_shape)
        for idx, noise_sample in enumerate(uncorrelated_noise):
            noise_sampe_fft = np.fft.fft(noise_sample)
            response_mag = np.sqrt(self.response)
            result_mag = response_mag * noise_sampe_fft

            inverse = np.fft.ifft(result_mag)
            rvs_noise_sample = np.real(inverse)
            ret[idx, :] = rvs_noise_sample

        return ret


NOISE_CLS = {
    'white': WhiteNoiseGenerator,
    'rvs': RVSNoiseGenerator
}
