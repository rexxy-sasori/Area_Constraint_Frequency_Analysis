import numpy as np

from IO.config import get_usr_config
from IO.result import DetectionResult
from core import detector
from core import freq_transform
from core import signal_generator


def run():
    usr_config = get_usr_config()
    np.random.seed(usr_config.seed)

    # prepare input signal
    input_signal_generator = signal_generator.InputSignalGenerator(usr_config.signal, usr_config.noise)
    input_signal, labels = input_signal_generator.get()

    # transform to frequency domain
    input_signal_freq_sq_mag, input_fft = freq_transform.transform_all(
        input_signal, usr_config.freq_transform_method, usr_config.signal)

    freq_detector = detector.HarmonicEstimator(usr_config.detection, input_signal_generator)
    roc, scores = freq_detector.get_roc(input_signal_freq_sq_mag, labels)

    # save result
    result = DetectionResult(
        roc, usr_config, input_signal_freq_sq_mag, input_signal_generator.input_snr, freq_detector, scores
    )

    result.save(usr_config.result_dir)


if __name__ == '__main__':
    run()
