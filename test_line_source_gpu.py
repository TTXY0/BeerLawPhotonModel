import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0_gpu

def gauss_density_pattern(xp, yp, amplitude, sigma):
    x, y = np.meshgrid(xp, yp)
    x0 = xp.mean()
    y0 = yp.mean()
    density = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return density

Lx = 50
Ly = 50

nx = 64
ny = 64

dx = Lx/nx
dy = Ly/ny

xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
mu = gauss_density_pattern(xc, yc, 50, Lx/10)

source_start = (-.7, -.5) #x,y
source_end = (-.3, .5)

source_direction = np.arccos( (source_end[0]-source_start[0]) / (source_end[1] - source_start[1]))
right_ray_direction = np.cos(-np.pi/2 + source_direction), np.sin(-np.pi/2 + source_direction)

ray_direction = right_ray_direction 


P0, a, mask = mu_to_p0_gpu.mu_to_p0_line_gpu(mu, (source_start), (source_end), ray_direction, dx/4, xc, yc)
plt.imshow(a, origin='lower')
plt.show()
plt.imshow(P0, origin='lower')
plt.show()
