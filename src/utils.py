import numpy as np
import matplotlib.pyplot as plt

# Check if data is successfully loaded
def show_images(X, labels, len_row=1):
    fig = plt.figure(figsize=(5,7))
    ax = []
    for i in range(X.shape[0]):
        ax.append(fig.add_subplot(X.shape[0], len_row, i+1))
        if i == 0:
            ax[-1].set_title('original')
        ax[-1].set_ylabel('label:'+labels[i])
        ax[-1].set_xticks([])
        ax[-1].set_yticks([])
        plt.imshow(X[i].reshape(5, 5), cmap='gray')    
    plt.show()

# Add noise to images
def add_noise_to_single_image(img, p=0.05):
    noise = np.random.uniform(0,1,len(img)) < p
    return np.where(noise, img * -1, img)

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))