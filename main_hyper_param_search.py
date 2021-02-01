import glob
import os
import time
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import tqdm
from tqdm import tqdm

from IO import config
from IO.result import DetectionResult
from core import detector
from core import freq_transform
from core import signal_generator


def run(usr_config):
    np.random.seed(usr_config.seed)

    # prepare input signal
    input_signal_generator = signal_generator.InputSignalGenerator(usr_config.signal, usr_config.noise)
    input_signal, labels = input_signal_generator.get()

    # transform to frequency domain
    input_signal_freq_sq_mag, input_fft = freq_transform.transform_all(
        input_signal, usr_config.freq_transform_method, usr_config.signal
    )
    input_signal_freq_sq_mag_half = input_signal_freq_sq_mag[:, 0:int(usr_config.signal.block_size / 2) + 1]

    freq_detector = detector.HarmonicEstimator(usr_config.detection, input_signal_generator)
    roc, scores = freq_detector.get_roc(input_signal_freq_sq_mag_half, labels)

    # save result
    result = DetectionResult(roc=roc, usr_configs=usr_config, scores=scores)
    result.save(usr_config.result_dir)


def experiment(usr_configs_template, hyper_param):
    usr_configs_template.signal.block_size = hyper_param.block_size
    usr_configs_template.signal.num_blocks_avg = hyper_param.num_blocks_avg
    usr_configs_template.signal.phases = [hyper_param.phi]
    usr_configs_template.signal.hop_size = usr_configs_template.signal.block_size * usr_configs_template.signal.num_blocks_avg

    usr_configs_template.noise.init_args.top = hyper_param.noise_level
    usr_configs_template.noise.init_args.steady_state = hyper_param.noise_level

    usr_configs_template.freq_transform_method.name = hyper_param.freq_transform_method
    usr_configs_template.detection.name = hyper_param.detection_method

    usr_configs_template.result_dir = os.path.join(
        '/home/hgeng4/pmsp/results',
        'Fmethod_{}'.format(hyper_param.freq_transform_method),
        'detection_{}'.format(hyper_param.detection_method),
        'phi_{}'.format(hyper_param.phi),
        'N_{}'.format(hyper_param.block_size),
        'L_{}'.format(hyper_param.num_blocks_avg),
        'inde_noise_level_{}'.format(hyper_param.noise_level),
    )

    N = usr_configs_template.signal.block_size
    fs = usr_configs_template.signal.fs
    test_k = np.linspace(3, 4, 101)
    test_f = test_k / N * fs

    if os.path.exists(usr_configs_template.result_dir, ):
        number_tar = glob.glob(os.path.join(usr_configs_template.result_dir, '*tar'))
        if len(number_tar) != 101:
            print(usr_configs_template.result_dir, '\n')
            # shutil.rmtree(usr_configs_template.result_dir)
            pass
        else:
            return

    for f in test_f:
        usr_configs_template.signal.freqs = [f]
        # run(usr_configs_template)


def multi_run_wrapper(args):
    return experiment(*args)


def run_single_machine(usr_configs_template, search_space, host_name=None):
    pool = Pool(processes=4)
    total_args = []
    for s in search_space:
        total_args.append([usr_configs_template, s])

    s = time.time()
    print('staring at', datetime.now())
    for _ in tqdm(pool.map(multi_run_wrapper, total_args), total=len(total_args)):
        pass
    e = time.time()
    print('end at', datetime.now())
    print('total time in second:', e - s)


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)

    usr_configs_template = config.parse_config('./yaml/template.yaml')
    search_space = config.parse_search_space('./yaml/search_space.yaml')
    total_args = [[usr_configs_template, s] for s in search_space]

    run_single_machine(usr_configs_template, search_space, host_name=None)
