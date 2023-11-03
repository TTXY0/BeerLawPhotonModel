import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0

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
#print(xc)

mu = gauss_density_pattern(xc, yc, 5, Lx/10) #Fix bug here
mu_background = 2

#For a ray_direction to the "right" of the light source

source_start = (-1.3, -.5) #x,y
source_end = (-1.3, .5)

source_direction = np.arccos( (source_end[0]-source_start[0]) / (source_end[1] - source_start[1]))
right_ray_direction = np.cos(-np.pi/2 + source_direction), np.sin(-np.pi/2 + source_direction)

ray_direction = right_ray_direction 



P0, a, fluence = mu_to_p0.mu_to_p0_line(mu, mu_background, (source_start), (source_end), ray_direction, dx/4, xc, yc)

plt.imshow(P0, origin='lower', cmap = 'gray')
plt.show()


plt.imshow(a, origin='lower', cmap = 'gray')
plt.show()
plt.imshow(fluence, origin='lower', cmap = 'gray')
plt.show()
