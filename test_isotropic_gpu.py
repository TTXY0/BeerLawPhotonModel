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
mu_background = 0.2

P0, a, fluence = mu_to_p0_gpu.mu_to_p0_gpu(mu, mu_background, (-Lx/2, -Ly/2), dx/4, xc, yc)

fig, ax = plt.subplots(1, 4, figsize=(12, 4))
ax[0].set_title("Attenuation Matrix")
ax[1].set_title("'a'")
ax[2].set_title("fluence")
ax[3].set_title("P0")

ax[0].imshow(mu, cmap = "gray")
ax[1].imshow(a, cmap = "gray")
ax[2].imshow(np.exp(-a), cmap = "gray")
ax[3].imshow(P0, cmap = "gray")



plt.show()