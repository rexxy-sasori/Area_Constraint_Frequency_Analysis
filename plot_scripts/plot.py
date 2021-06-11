import matplotlib.pyplot as plt
import numpy as np

sim_snrs = np.load('sim.npy')
eval_snrs = np.load('eval.npy')

num_bits = 9
num_level = 20

bs = np.arange(3, 3+num_bits)
nls = np.arange(-15, -15+num_level)

def linear2log(linear):
	return 10*np.log10(linear)

def plot_snr(sim_snrs, eval_snrs, nlidx, nl, nls):
	sim_snr = sim_snrs[:,nlidx, :]
	eval_snr = eval_snrs[:,nlidx, :]

	print(nlidx, -nl)
	print(linear2log(sim_snr))
	print(linear2log(eval_snr))

for nlidx, nl in enumerate(nls):
	plot_snr(sim_snrs, eval_snrs, nlidx=nlidx, nl=nl, nls=nls)


