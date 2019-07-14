from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()


x = np.arange(-10, 10, 0.1)

y = np.arange(-10, 10, 0.1)

x, y = np.meshgrid(x,y)

z = 1/20*x**2 + y**2

ax = Axes3D(fig)

ax.plot_surface(x,y,z)
plt.show()