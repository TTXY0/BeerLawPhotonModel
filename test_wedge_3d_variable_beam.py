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

source_start = np.array([-1, -1, -.25]) #vertical source
source_end = np.array([-1, -1, .25])

#ray direction is defined as the rotational angle around the z-axis in radians
ray_direction = np.pi / 4 #np.pi/10
theta = np.pi/4

I_row = create_gaussian_array(11, 5, 5)
I = np.array([I_row for _ in range(nz)])

P0, a, fluence = mu_to_p0.mu_to_p0_wedge_variable_beam_3d(mu, mu_background, source_start, source_end, ray_direction, theta, dx/2, xc, yc, zc, I)
X, Y, Z = np.meshgrid(xc, yc, zc)

matlab_filename = "wedge.mat"

# Create a dictionary to store the data with a variable name (e.g., 'volume_data')
data_dict = {'volume_data': P0}

# Save the dictionary as a .mat file
sio.savemat(matlab_filename, data_dict)

#P0
X, Y, Z = np.meshgrid(xc, yc, zc)
marker_size = 30 * P0 / np.max(P0)
P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
                                   mode='markers', 
                                   marker=dict(size=marker_size.flatten(),
                                               color=P0.flatten(),
                                               colorscale='gray',
                                               opacity=1,
                                               colorbar=dict(title='Value')
                                   ))
layout = go.Layout(scene=dict(aspectmode='data'), title = "P0")
fig0 = go.Figure(data=[P0_Scatter], layout=layout)
fig0.show()

# Alpha
marker_size = 30 * a / np.max(a)
P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
                                   mode='markers', 
                                   marker=dict(size=marker_size.flatten(),
                                               color=a.flatten(),
                                               colorscale='gray',
                                               opacity=1,
                                               colorbar=dict(title='Value')
                                   ))
layout = go.Layout(scene=dict(aspectmode='data'), title= "Alpha")
fig1 = go.Figure(data=[P0_Scatter], layout=layout)
fig1.show()

# Fluence
fluence = np.exp(-a)
marker_size = 30 * fluence / np.max(fluence)
P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
                                   mode='markers', 
                                   marker=dict(size=marker_size.flatten(),
                                               color=fluence.flatten(),
                                               colorscale='viridis',
                                               opacity=1,
                                               colorbar=dict(title='Value')
                                   ))
layout = go.Layout(scene=dict(aspectmode='data'), title= "Fluence")
fig2 = go.Figure(data=[P0_Scatter], layout=layout)
fig2.show()
