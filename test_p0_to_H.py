import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
import inverse_problem


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

#I_control = np.ones(11)
I = create_gaussian_array(11, 9, 5)
direction = 0
theta = np.pi / 10
source = (-10, 0)

#P0_original, a_original, fluence_original = mu_to_p0.mu_to_p0_cone_variable_beam(mu, mu_background, source, dx/4, xc, yc, theta, direction, I)
P0_original, a_original, fluence_original = mu_to_p0.mu_to_p0_cone_variable_beam(mu, mu_background, source, dx/4, xc, yc, theta, direction, I)
P0, a, fluence = mu_to_p0.mu_to_p0_cone(mu, mu_background, source, dx/4, xc, yc, theta, direction)
H = inverse_problem.p0_to_H(mu, P0, I, source, xc, yc, theta, direction)

y = np.dot(H, I)
P0_new = y.reshape(mu.shape[0], -1)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

ax[0].set_title("P0_new")
ax[1].set_title("P0_original")
# ax[2].set_title("fluence")
# ax[3].set_title("P0")

ax[0].imshow(P0_new, cmap = "gray")
ax[1].imshow(P0_original, cmap = "gray")
# ax[2].imshow(fluence, cmap = "gray")
# ax[3].imshow(P0, cmap = "gray")

print(np.allclose(P0_original, P0_new, atol= 0.00001))

plt.show()
