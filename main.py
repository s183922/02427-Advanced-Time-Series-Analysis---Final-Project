from MCL import MCL, get_prediction
from tqdm import tqdm
from matplotlib.animation import PillowWriter
from matplotlib.patches import Circle
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from interp import measure as measure_full
from functools import partial
from Navigation import Compass, Difference

Navigator = Difference
gif_title = 'Navigation with difference'
n_particles = 10000
uncertainty = 0.15
T = 250
sigma = 1
dt = 1
start_point = [30, 150]
true_pos = start_point
max_speed = 2
fog_visibility = 5
velocity = [0, 0]

data = loadmat('Bolmen.mat')

depth = np.array(data['Depth']).T

measure = partial(measure_full, data['xscale'][0], data['yscale'][0], depth)
writer = PillowWriter(fps=2)
animation_figure, animation_ax = plt.subplots()
writer.setup(animation_figure, f'{gif_title}.gif')

target = np.array([data['harbour'][0][0], data['harbour'][0][1]])

Navigator = Navigator(target)

X, Y = np.meshgrid(data['xscale'], data['yscale'])

xlim = data['xscale'][0][-1]
ylim = data['yscale'][0][-1]

particles = np.zeros((n_particles, 3))
particles[:, 0] = xlim*np.random.random_sample(n_particles)
particles[:, 1] = ylim*np.random.random_sample(n_particles)
particles[:, 2] = 1

animation_figure = plt.figure()

while np.linalg.norm(true_pos - target) > fog_visibility:
    print(f'{np.linalg.norm(true_pos - target):.2f}', end='\r')
    new_pos = np.clip(true_pos + np.random.normal(0, sigma, 2)*np.sqrt(dt) + velocity, [0, 0], [xlim, ylim])
    velocity = new_pos - true_pos
    true_pos = new_pos
    
    true_depth = measure(true_pos)
    noise = uncertainty*2*(np.random.random_sample()-0.5)*true_depth
    measurement = true_depth + noise
    particles = MCL(particles, measure, measurement, uncertainty, velocity, sigma, dt, xlim, ylim)
    pred, std = get_prediction(particles)
    velocity = Navigator.get_velocity(pred)
    velocity = max_speed*velocity/np.linalg.norm(velocity)
    

    animation_ax.clear()
    animation_ax.contour(X, Y, data['Depth'], levels=15)
    animation_ax.scatter(particles[:, 0], particles[:, 1], c='blue', s=0.1)
    animation_ax.scatter(true_pos[0], true_pos[1], c='black')
    animation_ax.scatter(pred[0], pred[1], c='red')
    animation_ax.scatter(target[0], target[1], c='purple')
    circle = Circle((pred[0], pred[1]), radius = std, alpha=0.5)
    animation_ax.add_patch(circle)
    animation_ax.set_title('Navigation with compass')
    writer.grab_frame()

writer.finish()