import glob
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_scores_hist_single(results, file_name):
    num_results = len(results)
    imgs = []
    for i in range(num_results):
        result = results[i]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(result.scores, bins=500)
        ax.grid()
        ax.set_ylim([0, 4000])
        ax.set_xlim([0, 1])
        ax.set_title('freq={} Hz'.format(result.usr_configs.signal.freqs[0]), fontsize=15)
        ax.tick_params('both', labelsize=25)
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(image)

    kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    imageio.mimsave(file_name, imgs, fps=2)


def plot_scores_hist(result_dir, identifier='*tar'):
    data_dir = os.path.join(result_dir, identifier)
    result_identifier = os.path.split(result_dir)[1]
    file_name = os.path.join('../plots', 'hist' + result_identifier + '.gif')
    result_paths = glob.glob(data_dir)

    results = []
    for p in result_paths:
        result = torch.load(p).get('result')
        results.append(result)

    results = sorted(results, key=lambda x: x.usr_configs.signal.freqs[0])

    plot_scores_hist_single(results, file_name)


if __name__ == '__main__':
    result_dirs = glob.glob('../results/*')
    for result_dir in result_dirs:
        print('plotting results in {}'.format(result_dir))
        plot_scores_hist(result_dir, '*tar')
