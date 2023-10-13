import numpy as np
import matplotlib.pyplot as plt
import mu_to_p0


def p0_to_H(mu, p0, I, source, xc, yc, theta, direction):
    xs, ys = source
    
    n = mu.shape[0]
    column =  p0.flatten()
    
    theta_start = direction - (theta/2)
    theta_end = direction + (theta/2)
    d_theta = theta / (len(I) -1)
    
    
    H = np.column_stack([column for _ in range(len(I))])
    
    for m in range( H.shape[0] ) : 
        
        i = m % n
        j = m // n
        print(i, j)
        
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

