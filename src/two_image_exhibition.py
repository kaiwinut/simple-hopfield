from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt
from utils import add_noise_to_single_image, show_images

""" Preparing data
"""
PATH = join(dirname(dirname(__file__)), 'data', 'alphabets.csv')
data = np.loadtxt(PATH, delimiter=',', dtype='unicode')
X = data[:2, :-1].astype(np.float32)
labels = data[:2, -1]

X_exhibit = []
labels_exitbit = []
iters = [25, 50, 100, 500, 1000]

""" Training a Hopfield network

1. Initialize weights
2. Randomly pick one neuron and update its value
3. Repeat step 2 until the result converges
"""
# Initialize weights with single pattern
W = np.dot(X.T, X)
W = np.where(np.eye(X.shape[1]) == 1, 0, W)

for img_idx in [0, 1]:
    # Add noise to image
    for noise in [0.05, 0.2, 0.5]:
        img_original = X[img_idx]
        X_exhibit.append(list(img_original))
        labels_exitbit.append(f"original")

        img_noisy = add_noise_to_single_image(img_original, noise)
        X_exhibit.append(list(img_noisy))
        labels_exitbit.append(f"noise:{noise:.2f}")
        threshold = np.zeros(img_noisy.shape[0])

        for it in range(1000):
            # Randomly pick one neuron
            idx = np.random.randint(0, len(img_noisy))
            # Update value
            out = np.dot(W[idx], img_noisy) - threshold[idx]
            img_noisy[idx] = 1 if out >= 0 else -1

            if (it+1) in iters:
                X_exhibit.append(list(img_noisy))
                labels_exitbit.append(f"iters:{it+1}")

""" Visualizing results
"""
X_exhibit = np.array(X_exhibit)
fig = plt.figure(figsize=(10,7))
ax = []
for i in range(X_exhibit.shape[0]):
    ax.append(fig.add_subplot(6, 7, i+1))
    if i//7 == 0 and i%7 != 1:
        ax[-1].set_title(labels_exitbit[i])
    elif i//7 == 0 and i%7 == 1:
        ax[-1].set_title('noisy')
    if i%7 == 0:
        ax[-1].set_ylabel(labels_exitbit[i+1])
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    plt.imshow(X_exhibit[i].reshape(5, 5), cmap='gray')   

IMAGE_PATH = join(dirname(dirname(__file__)), 'images', 'exhibition.png') 
plt.savefig(IMAGE_PATH)
plt.show()