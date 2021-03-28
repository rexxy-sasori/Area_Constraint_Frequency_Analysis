from datetime import datetime

import numpy as np

from IO import config
from core import signal_generator
from core import utils

B = 3


def sim_fx_dp_noise_model(noise_level, model_coeff_generator, Bx=B, Bw=B, Ba=B):
    configurations = config.parse_config('../yaml/example.yaml')

    noise_level_db = noise_level
    noise_level_linear = utils.db_to_linear(noise_level_db)

    configurations.signal.num_pos_decision = 1000
    configurations.noise.init_args.top = noise_level_linear
    configurations.noise.init_args.steady_state = noise_level_linear

    generator = signal_generator.InputSignalGenerator(configurations.signal, configurations.noise)
    signal_with_noise, _ = generator.get()

    configurations.noise.init_args.top = 0
    configurations.noise.init_args.steady_state = 0

    generator = signal_generator.InputSignalGenerator(configurations.signal, configurations.noise)
    signal_without_noise, _ = generator.get()

    noise_sample = signal_with_noise - signal_without_noise

    Nd = configurations.signal.num_pos_decision
    N = configurations.signal.block_size
    signal_with_noise = signal_with_noise[0:Nd, :, :]
    signal_without_noise = signal_without_noise[0:Nd, :, :]
    noise_sample = noise_sample[0:Nd, :, :]

    coeff_jdht = np.zeros((Nd, N, N))
    for i in range(Nd):
        coeff_jdht[i, :, :] = model_coeff_generator(N) / N

    var_jdht_coeff = np.var(coeff_jdht)
    signal_power = np.var(signal_without_noise)

    w_max = np.abs(coeff_jdht).max()
    qcoeff_jdht = utils.unisign_quant(coeff_jdht, Bw, w_max)
    deltaw = utils.get_delta(Bw, w_max)

    sigal_max = np.abs(signal_with_noise).max()
    qsignal_with_noise = utils.unisign_quant(signal_with_noise, Bx, sigal_max)
    deltax = utils.get_delta(Bx, sigal_max)

    result_noise = np.zeros(signal_without_noise.shape)
    result_signal_without_noise = np.zeros(signal_without_noise.shape)
    result_signal_with_noise = np.zeros(signal_without_noise.shape)
    qresult_signal_without_noise = np.zeros(signal_without_noise.shape)
    qresult_signal_with_noise = np.zeros(signal_without_noise.shape)

    start = datetime.now()
    for i in range(Nd):
        result_signal_without_noise[i, :, :] = signal_without_noise[i, :, :].dot(coeff_jdht[i, :, :].T)
        result_noise[i, :, :] = noise_sample[i, :, :].dot(coeff_jdht[i, :, :].T)
        result_signal_with_noise[i, :, :] = signal_with_noise[i, :, :].dot(coeff_jdht[i, :, :].T)
        qresult_signal_with_noise[i, :, :] = qsignal_with_noise[i, :, :].dot(qcoeff_jdht[i, :, :].T)

    result_max = result_signal_with_noise.max()
    deltay = utils.get_delta(Ba, result_max)
    qresult_signal_with_noise = utils.unisign_quant(qresult_signal_with_noise, Ba, result_max)
    end = datetime.now()

    total_dp_noise = qresult_signal_with_noise - result_signal_without_noise
    var_total_dp_noise = total_dp_noise.var()
    A = deltay ** 2 / 12
    B = N * var_jdht_coeff * noise_level_linear
    C = N / 12 * var_jdht_coeff * deltax ** 2
    D = N / 12 * signal_without_noise.var() * deltaw ** 2
    E = N / 12 * noise_level_linear * deltaw ** 2
    predicted_total_var = A + B + C + D + E

    predicted_signal_power = N * signal_power * var_jdht_coeff
    sim_signal_power = result_signal_without_noise.var()

    predicted_snr = predicted_signal_power / (A + B + C + D + E)
    sim_snr = sim_signal_power / var_total_dp_noise
    print('total time (s):', utils.format_float((end - start).total_seconds()),
          'Bx:', Bx, 'Bw:', Bw, 'Ba:', Ba,
          noise_level, 'dB:', 10 * np.log10(predicted_total_var / var_total_dp_noise),
          10 * np.log10(predicted_signal_power / sim_signal_power),
          10 * np.log10(predicted_snr / sim_snr))
