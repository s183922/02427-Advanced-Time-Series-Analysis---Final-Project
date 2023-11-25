from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from scipy.ndimage import laplace

colors = [(1,0,0,c) for c in np.linspace(0,1,100)]
cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=20)
cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)


class Map:
    def __init__(self, sigma = 1, noise_level = .15, initial_position = [0,0]) -> None:
        self.sigma = sigma
        self.noise_level = noise_level
        self.position = np.array(initial_position)

    def load_data(self, path: str):
        '''Loads the data from a .mat file and provides interpolation to a regular grid.'''
        data = loadmat(path)
        self.depth = data['Depth']
        self.harbour = data['harbour']
        xscale, yscale = data['xscale'], data['yscale']
        self.measure = RegularGridInterpolator((xscale[0], yscale[0]), self.depth.T)
        self.X, self.Y = np.meshgrid(xscale, yscale)

    def _get_depth(self):
        return self.measure(self.position)
    
    def measurement(self):
        '''Returns the measurement of the depth at the current position.'''
        return self._get_depth() * np.random.uniform(1-self.noise_level, 1+self.noise_level)

    def step(self, u = None):
        '''Steps the position according to the input u.'''
        self.position = self.position + u if u is not None else self.position
        self.position = self.position + np.random.normal(0, self.sigma, 2)
        self.position[0] = np.clip(self.position[0], 0, self.X.max())
        self.position[1] = np.clip(self.position[1], 0, self.Y.max())

    def plot_map(self, ax):
        '''Plots the map.'''
        ax.contourf(self.X, self.Y, self.depth, cmap = 'Blues_r')
        ax.contour(self.X, self.Y, self.depth, levels = [-0.5])
        ax.scatter(self.harbour[:,0], self.harbour[:,1], c = 'r', marker = 'x', label = 'Harbour')
        ax.legend()
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.axis('equal')
        ax.axis('tight')


class NavigationBayes(Map):
    def __init__(self, sigma = 1, noise_level = .15, initial_position = [0,0]) -> None:
        super().__init__(sigma, noise_level, initial_position)

    
    def locate(self, n_steps = 1000):
        '''Locates the position of the boat by taking n_steps steps and measuring the depth.'''
        self.measurements = np.zeros(n_steps)
        self.positions = np.zeros((n_steps, 2))
        self.posterior = np.ones_like(self.depth) / self.depth.size
        for i in range(n_steps):
            self.measurements[i] = self.measurement()
            self.positions[i] = self.position
            self.step()

        self.animation(self.positions, self.measurements)
        return self.measurements

    def likelihood(self, observation):
        '''Returns the likelihood of the observation given the map.'''
        l = (self.depth<=observation/(1 + self.noise_level)) & (observation/(1-self.noise_level)<=self.depth)
        return l
    
    def time_step(self, P):
        '''Time step of the Bayesian filter. Kolmogorov forward equation.'''
        L = laplace(P)
        P = P + 1/2 * self.sigma**2 * L 
        return P

    def update(self, observation):
        '''Bayesian filter. Returns the posterior distribution of the position given the observations.'''
        self.posterior = self.time_step(self.posterior)
        self.posterior *= self.likelihood(observation)
        self.posterior /= self.posterior.sum()
        return self.posterior

    def plot_likelihood(self, ax, observation):
        '''Plots the likelihood of the observation given the map.'''
        l = self.likelihood(observation)
        ax.contourf(self.X, self.Y, l, cmap = cmapred)    

    def animation(self, positions, measurements):
        '''Animates the boat taking n_steps steps and measuring the depth.'''
        fig, ax = plt.subplots()
        # self.plot_map(ax)
        ax.set_title('Navigation')

        def animate(i):
            ax.clear()
            self.plot_map(ax)
            # self.plot_likelihood(ax, measurements[i])
            p = self.update(measurements[i])
            ax.contourf(self.X, self.Y, p, cmap = cmapred)
            ax.scatter(positions[i,0], positions[i,1], c = 'k', marker = 'o', label = 'Position')
            MAP = np.unravel_index(p.argmax(), p.shape)
            ax.scatter(self.X[MAP], self.Y[MAP], c = 'blue', marker = '*', label = 'MAP')
            ax.annotate(f'Step: {i}', (0.05, 0.95), xycoords='axes fraction', fontsize=12,
                        horizontalalignment='left', verticalalignment='top')
            ax.legend()
        ani = animation.FuncAnimation(fig, animate, frames = len(measurements))
        writer = animation.PillowWriter(fps=20,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
        ani.save('navigation2.gif', writer=writer)
        plt.show()


if __name__ == '__main__':
    nav = NavigationBayes(initial_position=[40,120], sigma = 0.5, noise_level=0.15)
    nav.load_data('maps/Bolmen.mat')

    nav.locate()

