import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0


def gauss_density_pattern(xp, yp, zp, amplitude, sigma):
    x, y, z = np.meshgrid(xp, yp, zp)
    x0 = xp.mean()
    y0 = yp.mean()
    z0 = zp.mean()
    density = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
    density = np.transpose(density, (2, 1, 0)) #convert to zyx
    return density

Lx = 10
Ly = 10
Lz = 10

nx = 32
ny = 32
nz = 32

dx = Lx / nx
dy = Ly / ny
dz = Lz / nz

xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
zc = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, nz)

mu = gauss_density_pattern(xc, yc, zc, .5, Lx/10)
mu_background = .2

source_start = np.array([-10, 0, -2.5]) #vertical source
source_end = np.array([-10, 0, 2.5])

#ray direction is defined as the rotational angle around the z-axis in radians
ray_direction = 0 #np.pi/10
theta = np.pi/10

P0, a, fluence = mu_to_p0.mu_to_p0_wedge_3d(mu, mu_background, source_start, source_end, ray_direction, theta, dx/2, xc, yc, zc)
fig, ax = plt.subplots(3, 3, figsize=(12,7), sharex= True, sharey= True)
ax[0,0].set_title("Along Z axis", fontsize = 20)
ax[0,1].set_title("Along Y axis", fontsize = 20)
ax[0,2].set_title("Along X axis", fontsize = 20)

ax[0,0].imshow(np.max(P0, axis = 0), cmap = 'gray')
ax[0,1].imshow(np.max(P0, axis = 1), cmap = 'gray')
ax[0,2].imshow(np.max(P0, axis = 2), cmap = 'gray')

ax[1,0].imshow(np.max(a, axis = 0), cmap = 'gray')
ax[1,1].imshow(np.max(a, axis = 1), cmap = 'gray')
ax[1,2].imshow(np.max(a, axis = 2), cmap = 'gray')

ax[2,0].imshow(np.max(fluence, axis = 0), cmap = 'gray')
ax[2,1].imshow(np.max(fluence, axis = 1), cmap = 'gray')
ax[2,2].imshow(np.max(fluence, axis = 2), cmap = 'gray')

ax[0,0].set_ylabel("P0", fontsize = 20)
ax[1,0].set_ylabel("Alpha", fontsize = 20)
ax[2,0].set_ylabel("Fluence", fontsize = 20)


fig.tight_layout()
plt.savefig("wedge_MIPs")