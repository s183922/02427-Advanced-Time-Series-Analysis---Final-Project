import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt




def plot_map(path):
    map_data = loadmat(path)

    depth = map_data['Depth']
    home  = map_data['harbour'][0]
    xscale = map_data['xscale']
    yscale = map_data['yscale']
    print(home, depth[home[1], home[0]])
    X, Y = np.meshgrid(xscale, yscale)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot([1,1,-8], [1,1,0] , c='r')    
    ax.plot_surface(X, Y, depth, cmap='viridis', edgecolor='none', alpha = 0.5)
    plt.show()
plot_map('maps/Bolmen.mat')
