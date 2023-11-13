import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0
from scipy.sparse import lil_matrix, hstack, csr_matrix

# This is implemented with a 2d "wedge" source in mind
def p0_to_H_wedge(p0, I, source, xc, yc, theta, direction):
    xs, ys = source
    
    n = p0.shape[1]
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

def p0_to_sparseH_wedge(p0, I, source, xc, yc, theta, direction):
    xs, ys = source
    
    n = p0.shape[1]
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

def p0_to_H_cone(p0, I, source, xc, yc, zc, theta, direction_vector):
    xs, ys, zs = source
    
    n = p0.shape[2]
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

def p0_to_sparseH_cone(p0, I, source, xc, yc, zc, theta, direction_vector):

    xs, ys, zs = source
    n = p0.shape[2]
    column = p0.flatten()

    d_theta = theta / (len(I) - 1)

    H = lil_matrix((column.shape[0], I.shape[0]))
    # print(xc.shape, yc.shape, zc.shape)
    for m in range(H.shape[0]):
        i = m % n
        j = (m // n) % n
        k = m // (n * n)

        xi = xc[i]  # physical coordinates
        yi = yc[j]
        zi = zc[k]  # physical coordinates

        vector_shift = np.array([xi - xs, yi - ys, zi - zs])
        point_angle = np.arccos(   np.dot(direction_vector, vector_shift)  /  (np.linalg.norm(direction_vector) * np.linalg.norm(vector_shift)   )   )

        if point_angle <= theta / 2:
            outside_ray_i = int(np.ceil((point_angle) / d_theta))
            inside_ray_i = int(np.floor((point_angle) / d_theta))

            outside_ray_theta = outside_ray_i * d_theta
            inside_ray_theta = inside_ray_i * d_theta

            inside_ray_weight = abs(point_angle - outside_ray_theta) / d_theta
            outside_ray_weight = abs(point_angle - inside_ray_theta) / d_theta

            H[m, outside_ray_i] += column[m] * outside_ray_weight
            H[m, inside_ray_i] += column[m] * inside_ray_weight
        
    H = H.tocsr()
    return H # returns a shape (nx * ny * nz , len(I)) matrix

        
def p0_to_H_stackedCone(mu, mu_background, h, source_start, source_end, xp, yp, zp, direction_vector, theta, I):
    # H_stackedCone = np.zeros((mu.size, len(I) * len(zp)))
    xs, ys, zs = source_start
    xe, ye, ze = source_end
    dpz = zp[1] - zp[0]

    # if ze > zs:
    #     step = dpz
    # else:
    #     step = -dpz
    
    z_level = min(zs, ze)

    iteration = 0
    for i_z in range(len(zp)) : 
        z_level = zp[i_z]
        
        # if z_level != ze or z_level != ze: 

        #(min(zs, ze) <= z_level <= max(zs, ze)) and 
        if  (min(zs, ze) <= z_level <= max(zs, ze)):
            #print('new', z_level, (zs, ze))
            
            p0_single_cone, alpha, fluence = mu_to_p0.mu_to_p0_cone_3d(mu, mu_background, (xs, ys, z_level), h, xp, yp, zp, direction_vector, theta)
            H_cone = p0_to_sparseH_cone(p0_single_cone, I, (xs, ys, z_level), xp, yp, zp, theta, direction_vector)
            
            if iteration == 0 : 
                H_stack = H_cone
            else: 
                H_stack = hstack([H_stack, H_cone])
            

        else : 
            if iteration == 0 : 
                H_stack = csr_matrix((mu.size, len(I)), dtype=np.float64)
            else: 
                H_stack = hstack([H_stack, csr_matrix((mu.size, len(I)), dtype=np.float64)])
            
        iteration += 1

    return H_stack.toarray()