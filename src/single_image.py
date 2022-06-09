from os.path import join, dirname
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style('darkgrid')
from utils import similarity, add_noise_to_single_image

""" Preparing data
"""
PATH = join(dirname(dirname(__file__)), 'data', 'alphabets.csv')
data = np.loadtxt(PATH, delimiter=',', dtype='unicode')
X = data[:, :-1].astype(np.float32)
labels = data[:, -1]

""" Training a Hopfield network

1. Initialize weights
2. Randomly pick one neuron and update its value
3. Repeat step 2 until the result converges
"""
# Log
LOG_PATH = join(dirname(dirname(__file__)), 'data', 'single_image_results.csv')
log = 'label,noise,success,iterations,similarity\n'
TRIALS_PER_NOISE = 200

# Add noise to image
for img_idx in range(X.shape[0]):
    # Initialize weights with single pattern
    W = np.dot(X[img_idx].reshape(-1,1), X[img_idx].reshape(1,-1))
    W = np.where(np.eye(len(X[img_idx])) == 1, 0, W)
    for noise in np.arange(0.05, 0.20, 0.01):
        for trial in range(TRIALS_PER_NOISE):
            img_original = X[img_idx]
            img_noisy = add_noise_to_single_image(img_original, noise)
            threshold = np.zeros(img_noisy.shape[0])

            for it in range(1000):
                # print(f'iter {it+1} | errors {(img_noisy != img_original).sum()}')
                # End update if the image is identical to the original one
                if (img_noisy == img_original).all():
                    break
                # Randomly pick one neuron
                idx = np.random.randint(0, len(img_noisy))
                # Update value
                out = np.dot(W[idx], img_noisy) - threshold[idx]
                if out > 0:
                    img_noisy[idx] = 1
                elif out < 0:
                    img_noisy[idx] = -1

            log += f'{labels[img_idx]},{noise:.2f},{(img_noisy == img_original).all()},{it},{similarity(img_noisy, img_original)}\n'

with open(LOG_PATH, 'w') as f:
    f.write(log)

""" Visualizing results
"""

IMAGE_PATH_ACC = join(dirname(dirname(__file__)), 'images', 'single_image_results_acc.png')
IMAGE_PATH_SIM = join(dirname(dirname(__file__)), 'images', 'single_image_results_sim.png')
IMAGE_PATH_ITER = join(dirname(dirname(__file__)), 'images', 'single_image_results_iter.png')
results = pd.read_csv(LOG_PATH)
sns.relplot(data=results, x='noise', y='success', col='label', col_wrap=3, kind='line', hue='label', markers=True, dashes=False, style='label')
plt.savefig(IMAGE_PATH_ACC)
plt.clf()
sns.relplot(data=results, x='noise', y='similarity', col='label', col_wrap=3, kind='line', hue='label', markers=True, dashes=False, style='label')
plt.savefig(IMAGE_PATH_SIM)
plt.clf()
sns.relplot(data=results, x='noise', y='iterations', col='label', col_wrap=3, kind='line', hue='label', markers=True, dashes=False, style='label')
plt.savefig(IMAGE_PATH_ITER)
plt.clf()