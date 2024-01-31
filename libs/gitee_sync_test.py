import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# 创建3D坐标轴
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# 定义平面方程
xx = np.arange(-5, 5, 0.25)
yy = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(xx, yy)
Z = np.ones_like(X)*(-1)
Val = X**2+Y**2 #Change Here
# 绘制平面并设置颜色值
cmap = plt.get_cmap('rainbow')
norm = plt.Normalize(Val.min(), Val.max())
colors = cmap(norm(Val))
ax.plot_surface(X, Y, Z, facecolors=colors)
yy = np.arange(-5, 5, 0.25)
zz = np.arange(-5, 5, 0.25)
Y, Z = np.meshgrid(yy, zz)
X = np.ones_like(yy)
Val = X**2+Y**2 #Change Here
# 绘制平面并设置颜色值
cmap = plt.get_cmap('rainbow')
norm = plt.Normalize(Val.min(), Val.max())
colors = cmap(norm(Val))
ax.plot_surface(X, Y, Z, facecolors=colors)
plt.show()
