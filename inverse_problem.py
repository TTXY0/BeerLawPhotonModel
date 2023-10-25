import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
from scipy.sparse import lil_matrix

# This is implemented with a 2d "wedge" source in mind
def p0_to_H_wedge(mu, p0, I, source, xc, yc, theta, direction):
    xs, ys = source
    
    n = mu.shape[0]
    column =  p0.flatten()
    
    theta_start = direction - (theta/2)
    theta_end = direction + (theta/2)
    d_theta = theta / (len(I) -1)
    
    
    H = np.zeros( (column.shape[0], I.shape[0]))
    
    for m in range( H.shape[0] ) : 
        
        i = m % n
        j = m // n
        #print(i, j)
        
        xi = xc[i] #physical coordinates
        yi = yc[j]
        
        point_angle = np.arctan2 (yi - ys, xi - xs)
        
        if theta_start <= point_angle <= theta_end:
            clockwise_ray_i = int(np.ceil((point_angle - theta_start) / d_theta))
            countercw_ray_i = int(np.floor((point_angle - theta_start) / d_theta))
            
            clockwise_ray_theta = theta_start + (clockwise_ray_i * d_theta)
            countercw_ray_theta = theta_start + (countercw_ray_i * d_theta)
            
            clockwise_weight = abs(point_angle - countercw_ray_theta) / d_theta
            counterclockwise_weight = 1 - clockwise_weight
            
            H[m, clockwise_ray_i] += column[m] * clockwise_weight
            H[m, countercw_ray_i] += column[m] * counterclockwise_weight

    return H

def p0_to_sparseH_wedge(mu, p0, I, source, xc, yc, theta, direction):
    xs, ys = source
    
    n = mu.shape[0]
    column =  p0.flatten()
    
    theta_start = direction - (theta/2)
    theta_end = direction + (theta/2)
    d_theta = theta / (len(I) -1)
    
    
    H = lil_matrix((column.shape[0], I.shape[0]))
    
    for m in range( H.shape[0] ) : 
        
        i = m % n
        j = m // n
        
        xi = xc[i] #physical coordinates
        yi = yc[j]
        
        point_angle = np.arctan2 (yi - ys, xi - xs)
        
        if theta_start <= point_angle <= theta_end:
            clockwise_ray_i = int(np.ceil((point_angle - theta_start) / d_theta))
            countercw_ray_i = int(np.floor((point_angle - theta_start) / d_theta))
            
            clockwise_ray_theta = theta_start + (clockwise_ray_i * d_theta)
            countercw_ray_theta = theta_start + (countercw_ray_i * d_theta)
            
            clockwise_weight = abs(point_angle - countercw_ray_theta) / d_theta
            counterclockwise_weight = 1 - clockwise_weight
            
            H[m, clockwise_ray_i] += column[m] * clockwise_weight
            H[m, countercw_ray_i] += column[m] * counterclockwise_weight
    H = H.tocsr()
    return H

def p0_to_H_cone(mu, p0, I, source, xc, yc, zc, theta, direction_vector):
    xs, ys, zs = source
    
    n = mu.shape[0]
    column =  p0.flatten()
    
    dim_x , dim_y  = (column.shape[0], I.shape[0])
    H = np.zeros( (dim_x , dim_y))
    d_theta = theta / (len(I) -1)
    
    for m in range( H.shape[0] ) : 
        
        i = m % n
        j = (m // n) % n
        k = m // (n * n)
        
        xi = xc[i] #physical coordinates
        yi = yc[j]
        zi = zc[k] #physical coordinates
        
        vector_shift = np.array([xi - xs, yi - ys, zi - zs])
        point_angle = np.arccos(   np.dot(direction_vector, vector_shift)  /  (np.linalg.norm(direction_vector) * np.linalg.norm(vector_shift)   )   )

        if  point_angle <= theta:
            outside_ray_i = int(np.ceil((point_angle) / d_theta))
            inside_ray_i = int(np.floor((point_angle) / d_theta))
            
            outside_ray_theta = outside_ray_i * d_theta
            inside_ray_theta = inside_ray_i * d_theta
            
            inside_ray_weight = abs(point_angle - outside_ray_theta) / d_theta
            outside_ray_weight = 1 - inside_ray_weight
            
            H[m, outside_ray_i] += column[m] * outside_ray_weight
            H[m, inside_ray_i] += column[m] * inside_ray_weight

    return H

def p0_to_sparseH_cone(mu, p0, I, source, xc, yc, zc, theta, direction_vector):

    xs, ys, zs = source
    n = mu.shape[0]
    column = p0.flatten()

    d_theta = theta / (len(I) - 1)

    H = lil_matrix((column.shape[0], I.shape[0]))

    for m in range(H.shape[0]):
        i = m % n
        j = (m // n) % n
        k = m // (n * n)

        xi = xc[i]  # physical coordinates
        yi = yc[j]
        zi = zc[k]  # physical coordinates

        vector_shift = np.array([xi - xs, yi - ys, zi - zs])
        point_angle = np.arccos(
            np.dot(direction_vector, vector_shift)
            / (np.linalg.norm(direction_vector) * np.linalg.norm(vector_shift))
        )

        if point_angle <= theta:
            outside_ray_i = int(np.ceil((point_angle) / d_theta))
            inside_ray_i = int(np.floor((point_angle) / d_theta))

            outside_ray_theta = outside_ray_i * d_theta
            inside_ray_theta = inside_ray_i * d_theta

            inside_ray_weight = abs(point_angle - outside_ray_theta) / d_theta
            outside_ray_weight = 1 - inside_ray_weight

            H[m, outside_ray_i] += column[m] * outside_ray_weight
            H[m, inside_ray_i] += column[m] * inside_ray_weight

    H = H.tocsr()
    return H