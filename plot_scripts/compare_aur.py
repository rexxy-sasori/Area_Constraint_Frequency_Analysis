import glob
import os

import matplotlib.pyplot as plt
import torch
from sklearn import metrics

L = 'L_10'
phi = 'phi_0.7853981633974483'
dirs = {
    'DFT': '/home/hgeng4/pmsp/results/Fmethod_fft/detection_ml/' + phi + '/N_16/' + L + '/inde_noise_level_0.5',
    'DHT': '/home/hgeng4/pmsp/results/Fmethod_fht/detection_ml/' + phi + '/N_16/' + L + '/inde_noise_level_0.5',
    'J-DHT': '/home/hgeng4/pmsp/results/Fmethod_fht_jitter/detection_ml/' + phi + '/N_16/' + L + '/inde_noise_level_0.5',
    'SJ-DHT': '/home/hgeng4/pmsp/results/Fmethod_fht_jitter/detection_single/' + phi + '/N_16/' + L + '/inde_noise_level_0.5',
    'D-DHT': '/home/hgeng4/pmsp/results/Fmethod_fht_ditter/detection_ml/' + phi + '/N_16/' + L + '/inde_noise_level_0.5'
}

color = {
    'DFT': u'#1f77b4',
    'DHT': u'#2ca02c',
    'J-DHT': u'#ff7f0e',
    'D-DHT': 'indigo',
    'SJ-DHT': u'#17becf',
}

results_dict = {
    k: [] for k in dirs.keys()
}

plot_dir = os.path.join('/home/hgeng4/pmsp/results/plots', 'aur')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

plt.figure(figsize=(10, 5))

for name in results_dict.keys():
    results_paths = glob.glob(os.path.join(dirs[name], '*tar'))
    results = []
    for p in results_paths:
        result = torch.load(p).get('result')
        results.append(result)
    print('collected {} results @ {}'.format(len(results), dirs[name]))
    results_dict[name] = sorted(results, key=lambda x: x.usr_configs.signal.freqs[0])

    areas = []
    freqs = []
    ks = []
    for idx, result in enumerate(results):
        area = metrics.auc(result.roc.fpr, result.roc.tpr)
        areas.append(area)

        f0 = result.usr_configs.signal.freqs[0]
        k0 = result.usr_configs.signal.freqs[0] * result.usr_configs.signal.block_size / result.usr_configs.signal.fs

        freqs.append(float(f0))
        ks.append(k0)
    plt.plot(ks, areas, marker='o', markersize=5, label=name, c=color[name])

plt.xlabel('$k_o$', fontsize=15)
plt.ylabel('AUR', fontsize=15)
plt.grid()
plt.legend(fontsize=15)
plt.tick_params('both', labelsize=15)
plt.savefig(os.path.join(plot_dir, 'aur.png'))
plt.clf()
plt.close()
