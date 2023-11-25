import numpy as np
from numba import njit


def motion_update(x, u, sigma, dt, xlim, ylim):
    noise = np.random.normal(0, sigma, 2) * np.sqrt(dt)
    x[0] += u[0]*dt + noise[0]
    x[1] += u[1]*dt + noise[1]
    x = np.clip(x, np.zeros(2), np.array([xlim, ylim]))
    return x


def sensor_update(depth, measurement, uncertainty):
    llim = measurement/(1 + uncertainty)
    ulim = measurement/(1-uncertainty)
    return ((-depth>=-llim) & (-ulim>=-depth))/(llim-ulim)

def MCL(x, measure, measurement, uncertainty, u, sigma, dt, xlim, ylim):
    n_particles = x.shape[0]

    for particle in range(n_particles):
        x[particle, :2] = motion_update(x[particle, :2], u, sigma, dt, xlim, ylim)
        x[particle, 2] = sensor_update(measure(x[particle, :2]), measurement, uncertainty)
    
    normalization = np.sum(x[:, 2])
    if normalization == 0:
        print('No likely state found. Setting all to be equally likely.')
        w = np.ones(n_particles)/n_particles
    else:
        w = x[:, 2]/normalization
    new_particles = np.random.choice(np.arange(n_particles), size=n_particles, replace=True, p=w)

    return x[new_particles, :]


def get_prediction(particles):
    x = np.mean(particles[:, 0])
    y = np.mean(particles[:, 1])
    std = np.mean(np.sqrt((particles[:, 0] - x)**2 + (particles[:, 1] - y)**2))
    return np.array([x, y]), std