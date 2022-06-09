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
original_labels = data[:, -1]

# Log
LOG_PATH = join(dirname(dirname(__file__)), 'data', 'two_image_results.csv')
log = 'n_patterns,memorized_patterns,label,noise,success,iterations,similarity\n'
TRIALS_PER_NOISE = 200

for i in range(data.shape[0]):
    """ Training a Hopfield network

    1. Initialize weights
    2. Randomly pick one neuron and update its value
    3. Repeat step 2 until the result converges
    """

    # Add noise to image
    n = 2
    for noise in np.arange(0.0, 1.0, 0.05):
        for trial in range(TRIALS_PER_NOISE):
            # Change the combination of images
            X = data[:, :-1].astype(np.float32)
            labels = data[:, -1]
            random_order = np.random.permutation(X.shape[0])
            X = X[random_order]
            labels = labels[random_order]
            img_idx = np.argwhere(labels == original_labels[i])[0][0]
            X[0], X[img_idx] = X[img_idx], X[0]
            labels[0], labels[img_idx] = labels[img_idx], labels[0]
            # Initialize weights with single pattern
            W = np.dot(X[:n, :].T, X[:n, :])
            W = np.where(np.eye(X.shape[1]) == 1, 0, W)

            img_original = X[0]
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

            log += f'{n},{labels[:n]},{original_labels[i]},{noise:.2f},{(img_noisy == img_original).all()},{it},{similarity(img_noisy, img_original)}\n'

with open(LOG_PATH, 'w') as f:
    f.write(log)

""" Visualizing results
"""

IMAGE_PATH_ACC = join(dirname(dirname(__file__)), 'images', 'two_image_results_acc.png')
IMAGE_PATH_SIM = join(dirname(dirname(__file__)), 'images', 'two_image_results_sim.png')
IMAGE_PATH_ITER = join(dirname(dirname(__file__)), 'images', 'two_image_results_iter.png')
results = pd.read_csv(LOG_PATH)

sns.relplot(data=results, x='noise', y='success', col='label', col_wrap=3, kind='line', hue='label', palette='deep', markers=True, dashes=False, style='n_patterns')
plt.savefig(IMAGE_PATH_ACC)
plt.clf()
sns.relplot(data=results, x='noise', y='similarity', col='label', col_wrap=3, kind='line', hue='label', palette='deep', markers=True, dashes=False, style='n_patterns')
plt.savefig(IMAGE_PATH_SIM)
plt.clf()
sns.relplot(data=results, x='noise', y='iterations', col='label', col_wrap=3, kind='line', hue='label', palette='deep', markers=True, dashes=False, style='n_patterns')
plt.savefig(IMAGE_PATH_ITER)