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
# source_end = (-.95, -.5) #

# source_len = (.5)
# source_direction = np.pi/4 #direction cant be 0 or pi (180)

source_direction = np.arccos( (source_end[0]-source_start[0]) / (source_end[1] - source_start[1]))
right_ray_direction = np.cos(-np.pi/2 + source_direction), np.sin(-np.pi/2 + source_direction)
print(source_direction, right_ray_direction)

ray_direction = right_ray_direction 


print(right_ray_direction)
P0, a, fluence = mu_to_p0.mu_to_p0_line(mu, mu_background, (source_start), (source_end), ray_direction, dx/4, xc, yc)


# print(mu.shape, xc.shape, yc.shape)
# print(a[0,0])
# print(a[ny//2, nx//2])

plt.imshow(P0, origin='lower', cmap = 'gray')
plt.show()


plt.imshow(a, origin='lower', cmap = 'gray')
plt.show()
plt.imshow(fluence, origin='lower', cmap = 'gray')
plt.show()

#fig, ax = plt.subplots(1, 4, figsize=(12, 4))
#ax[1].scatter(pixelsx[nx//2, ny//2], pixelsy[nx//2., ny//2])


# ax[0].set_title("Attenuation Matrix")
# ax[1].set_title("'a'")
# ax[2].set_title("fluence")
# ax[3].set_title("P0")

# ax[0].imshow(mu, cmap = "gray")
# ax[1].imshow(a, cmap = "gray")
# ax[2].imshow(np.exp(-a), cmap = "gray")
# #ax[3].imshow(mask, cmap = 'gray')
# ax[3].imshow(P0, cmap = "gray")
# plt.show()