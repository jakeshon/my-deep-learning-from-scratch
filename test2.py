from matplotlib.image import imread
import matplotlib.pyplot as plt

img = imread('./dataset/lena.png')

plt.imshow(img)
plt.show()

