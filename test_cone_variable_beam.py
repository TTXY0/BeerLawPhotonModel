import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0

def gauss_density_pattern(xp, yp, amplitude, sigma):
    x, y = np.meshgrid(xp, yp)
    x0 = xp.mean()
    y0 = yp.mean()
    density = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return density
def create_gaussian_array(size, center, sigma): # for creating I
    x = np.arange(size)
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    return gaussian

Lx = 10
Ly = 10

nx = 64
ny = 64

dx = Lx/nx
dy = Ly/ny

xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)

mu = gauss_density_pattern(xc, yc, .50, Lx/10)
mu_background = .20

I = create_gaussian_array(11, 5, 5)
theta = np.pi/6
direction = np.pi/4
P0, a, fluence = mu_to_p0.mu_to_p0_cone_variable_beam(mu, mu_background, (-Lx/2, -Ly/2), dx/4, xc, yc, theta, direction, I)

fig, ax = plt.subplots(1, 5, figsize=(12, 4))

ax[0].set_title("Mask")
ax[1].set_title("Attenuation Matrix")
ax[2].set_title("'a'")
ax[3].set_title("fluence")
ax[4].set_title("P0")

ax[1].imshow(mu, cmap = "gray")
ax[2].imshow(a, cmap = "gray")
ax[3].imshow(fluence, cmap = "gray")
ax[4].imshow(P0, cmap = "gray")



plt.savefig("test1")