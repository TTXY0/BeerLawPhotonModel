import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0


def gauss_density_pattern(xp, yp, zp, amplitude, sigma):
    x, y, z = np.meshgrid(xp, yp, zp)
    x0 = xp.mean()
    y0 = yp.mean()
    z0 = zp.mean()
    density = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
    return density

def create_gaussian_array(size, center, sigma): # for creating I
    x = np.arange(size)
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return gaussian

Lx = 10
Ly = 10
Lz = 10

nx = 30
ny = 30
nz = 30

dx = Lx / nx
dy = Ly / ny
dz = Lz / nz

xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
zc = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, nz)

mu = gauss_density_pattern(xc, yc, zc, .5, Lx/10)
mu_background = .2

source = (-10, 0, 0)

#ray direction is defined as the rotational angle around the z-axis in radians
ray_direction = 0
theta = np.pi/10
direction_vector = [1,0,0]

I = create_gaussian_array(11, 5, 5)

print(mu.shape[0], mu.shape[1], mu.shape[2])
print(zc.shape[0], yc.shape[0], xc.shape[0])

P0, a, fluence = mu_to_p0.mu_to_p0_cone_variable_beam_3d(mu, mu_background, source, dx/2, xc, yc, zc, direction_vector, theta, I)
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
plt.savefig("variable_beamMIPS")