import numpy as np
import torch
from scipy.spatial import distance
from scipy.stats import entropy


def generate_data_SingleBatch(num_mode=100, radius=24, center=(0, 0), sigma=0.01, batchSize=64):
    num_data_per_class = int(np.ceil(batchSize/num_mode))
    total_data = {}

    t = np.linspace(0, 2*np.pi, num_mode+1)
    t = t[:-1]
    x = np.cos(t)*radius + center[0]
    y = np.sin(t)*radius + center[1]

    modes = np.vstack([x, y]).T

    for idx, mode in enumerate(modes):
        x = np.random.normal(mode[0], sigma, num_data_per_class)
        y = np.random.normal(mode[1], sigma, num_data_per_class)
        total_data[idx] = np.vstack([x, y]).T

    all_points = np.vstack([values for values in total_data.values()])
    all_points = np.random.permutation(all_points)[0:batchSize]
    return torch.from_numpy(all_points).float()


def loglikelihood(data, num_mode=100, radius=24, center=(0, 0)):
    t = np.linspace(0, 2*np.pi, num_mode+1)
    t = t[:-1]
    x = np.cos(t)*radius + center[0]
    y = np.sin(t)*radius + center[1]

    modes = np.vstack([x, y]).T
    q = np.ones(num_mode) / num_mode

    mat = distance.cdist(data, modes)
    prob = np.bincount(np.argmin(mat, axis=1), minlength=num_mode) / len(data)

    # find the entropy
    try:
        toReturn =  entropy(q,prob,base=2)
    except:
        print('Got some Error, return toReturn=-0.1')
        toReturn = -0.1
    return toReturn