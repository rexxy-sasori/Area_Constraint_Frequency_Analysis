from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(100)
y = np.arange(100)
plt.plot(x,y)
plt.xlabel(r'$SNR_T$'+'(dB)', fontsize=15)
plt.ylabel('$p_{tp}$', fontsize=15)
plt.show()