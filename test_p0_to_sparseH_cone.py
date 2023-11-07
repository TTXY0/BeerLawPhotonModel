import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
import inverse_problem


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

dx = Lx/nx
dy = Ly/ny
dz = Lz/nz


xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
zc = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, nz)

mu = gauss_density_pattern(xc, yc, zc, .5, Lx/10)
mu_background = .20

I = create_gaussian_array(11, 9, 5) #create assymetric gaussian array to test
theta = np.pi / 10
source = (-10, 0, 0)
direction_vector = [1,0,0]


P0_original, a_original, fluence_original = mu_to_p0.mu_to_p0_cone_variable_beam_3d(mu, mu_background, source, dx/2, xc, yc, zc, direction_vector, theta, I)
P0, a, fluence = mu_to_p0.mu_to_p0_cone_3d(mu, mu_background, source, dx/2, xc, yc, zc, direction_vector, theta)
sparse_H = inverse_problem.p0_to_sparseH_cone(P0, I, source, xc, yc, zc, theta, direction_vector)
print(sparse_H.shape)

y = sparse_H.dot(I)
P0_new = y.reshape(mu.shape[0], mu.shape[0], -1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].set_title("P0_new")
ax[1].set_title("P0_original")

ax[0].imshow(np.max(P0_new, axis = 1), cmap = "gray")
ax[1].imshow(np.max(P0_original, axis = 1), cmap = "gray")

print(np.allclose(P0_original, P0_new, atol= 0.00001))

plt.show()
