import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
import plotly.graph_objs as go
import scipy.io as sio


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

nx = 100
ny = 100
nz = 100

dx = Lx / nx
dy = Ly / ny
dz = Lz / nz

xc = np.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
yc = np.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
zc = np.linspace(-Lz/2 + dz/2, Lz/2 - dz/2, nz)

mu = gauss_density_pattern(xc, yc, zc, .5, Lx/10)
mu_background = .2

source_start = np.array([-10, 0, -3]) #vertical source
source_end = np.array([-10, 0, 3])

#ray direction is defined as the rotational angle around the z-axis in radians
ray_direction = 0
theta = np.pi/10

I_row = create_gaussian_array(11, 5, 5)
# I_row = np.ones(11)
I = np.array([I_row for _ in range(nz)])

print(mu.shape[0], mu.shape[1], mu.shape[2])
print(zc.shape[0], yc.shape[0], xc.shape[0])

P0, a, fluence = mu_to_p0.mu_to_p0_wedge_variable_beam_3d(mu, mu_background, source_start, source_end, ray_direction, theta, dx/2, xc, yc, zc, I)
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
# X, Y, Z = np.meshgrid(xc, yc, zc)

matlab_filename = "p0.mat"

# Create a dictionary to store the data with a variable name (e.g., 'volume_data')
data_dict = {'volume_data': P0}

# Save the dictionary as a .mat file
sio.savemat(matlab_filename, data_dict)

#P0
X, Y, Z = np.meshgrid(xc, yc, zc)
# marker_size = 30 * P0 / np.max(P0)
# P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
#                                    mode='markers', 
#                                    marker=dict(size=marker_size.flatten(),
#                                                color=P0.flatten(),
#                                                colorscale='gray',
#                                                opacity=1,
#                                                colorbar=dict(title='Value')
#                                    ))
# layout = go.Layout(scene=dict(aspectmode='data'), title = "P0")
# fig0 = go.Figure(data=[P0_Scatter], layout=layout)
# fig0.show()

# # Alpha
# marker_size = 30 * a / np.max(a)
# P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
#                                    mode='markers', 
#                                    marker=dict(size=marker_size.flatten(),
#                                                color=a.flatten(),
#                                                colorscale='gray',
#                                                opacity=1,
#                                                colorbar=dict(title='Value')
#                                    ))
# layout = go.Layout(scene=dict(aspectmode='data'), title= "Alpha")
# fig1 = go.Figure(data=[P0_Scatter], layout=layout)
# fig1.show()

# Fluence
# print(np.max(mask, axis = 1))
# marker_size = 30 * mask / np.max(mask)
# P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
#                                    mode='markers', 
#                                    marker=dict(size=marker_size.flatten(),
#                                                color=mask.flatten(),
#                                                colorscale='viridis',
#                                                opacity=1,
#                                                colorbar=dict(title='Value')
#                                    ))
# layout = go.Layout(scene=dict(aspectmode='data'), title= "mask")
# fig2 = go.Figure(data=[P0_Scatter], layout=layout)
# fig2.show()
