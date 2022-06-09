from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt

""" Preparing data
"""
PATH = join(dirname(dirname(__file__)), 'data', 'alphabets.csv')
data = np.loadtxt(PATH, delimiter=',', dtype='unicode')
X = data[:, :-1].astype(np.float32)
labels = data[:, -1]

""" Visualizing alphabets
"""
fig = plt.figure(figsize=(10,2))
ax = []
for i in range(X.shape[0]):
    ax.append(fig.add_subplot(1, 6, i+1))
    ax[-1].set_title(labels[i])
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    plt.imshow(X[i].reshape(5, 5), cmap='gray')  

IMAGE_PATH = join(dirname(dirname(__file__)), 'images', 'alphabet_exhibition.png') 
plt.savefig(IMAGE_PATH)
plt.show()