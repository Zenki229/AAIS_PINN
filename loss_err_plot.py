import numpy as np
import matplotlib.pyplot as plt
data = np.load('./results/Poisson2D1Peak/Poisson2D1Peak_Uni_resample_19076082_500e-pp1/train/loss_err.npy')
fig, ax = plt.subplots(layout='constrained', figsize=(19.2, 4.8))
ax.semilogy(np.array(data[:, 0]))
ax.axvline(x=5100, color='black', linestyle='--', linewidth=1)
plt.show()
