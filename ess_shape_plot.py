import numpy as np
import matplotlib.pyplot as plt
data3 = np.load('./results/Poisson2D1Peak/Poisson2D1Peak_AAIS_g_resample_10000e/train/ess_shape.npy')
data4 = np.load('./results/Poisson2D1Peak/Poisson2D1Peak_AAIS_t_resample_10000e/train/ess_shape.npy')
fig, ax = plt.subplots(layout='constrained', figsize=(6.4, 4.8))
ax.plot(np.array(data3[:, 0]), marker='o', label='AAIS-g')
ax.plot(np.array(data4[:, 0]), marker='o', label='AAIS-t')
# ax.axhline(y=0.03, color='black', linestyle='--', linewidth=1.0)
# ax.set_ylim(1e-2, 1e4)
ax.set_xticks(ticks=[0,3,6,9], labels=['1', '4', '7', '10'])
# ax.set_yticks(ticks=[1e-2,0.03,1e-1,1e0,1e1,1e2,1e3,1e4],labels=[f'$10^{-2}$','0.03','$10^{-1}$','$10^0$','$10^1$','$10^2$','$10^3$','$10^4$'])
ax.legend(loc='upper right')
fig.savefig('./results/Poisson2D1Peak/resample_10000e_ess.png', dpi=300)
plt.show()
