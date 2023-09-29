import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
import plotly.graph_objs as go


def gauss_density_pattern(xp, yp, zp, amplitude, sigma):
    x, y, z = np.meshgrid(xp, yp, zp)
    x0 = xp.mean()
    y0 = yp.mean()
    z0 = zp.mean()
    density = amplitude * np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * sigma**2))
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

source = (-Lx/2, -Ly/2, -Lz/2)

P0, a, fluence = mu_to_p0.mu_to_p0_cone_3d(mu, mu_background, source, dx/2, xc, yc, zc, [1,1,1], np.pi/10)

# # #P0
# X, Y, Z = np.meshgrid(xc, yc, zc)
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

# #Alpha
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

# #Fluence
# marker_size = 30 * fluence / np.max(fluence)
# P0_Scatter = go.Scatter3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
#                                    mode='markers', 
#                                    marker=dict(size=marker_size.flatten(),
#                                                color=fluence.flatten(),
#                                                colorscale='viridis',
#                                                opacity=1,
#                                                colorbar=dict(title='Value')
#                                    ))
# layout = go.Layout(scene=dict(aspectmode='data'), title= "Fluence")
# fig2 = go.Figure(data=[P0_Scatter], layout=layout)
# fig2.show()