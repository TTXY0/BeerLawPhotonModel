import numpy as np
def mu_to_p0(mu, source, h): #mu is an array of attenuation coefficients, source is a tuple containing the x and y coordinates, h is the spacing between sampling points along the ray
    xs, ys = source
    P = np.zeros_like(mu)
    
    for yi in range(mu.shape[0]):
        for xi in range(mu.shape[1]):
            m = (yi - ys) / (xi - xs)
            b = ys - m * xs
            
            d = ((xi-xs)**2 + (yi-ys)**2) **0.5 #euclidean distance between source and target
            n = int(d/h) + 1 # of discrete point
            
            dx =  (xi - xs) / (n - 1) #Fix? There are points outside the image dimensions
            dy = (yi - ys) / (n - 1)
            
            for point_i in range(n):
                x = int(np.floor(xs + (point_i * dx))) #The pixel at which the point is located
                y = int(np.floor(ys + (point_i * dy))) # ^

                if 0 <= x < mu.shape[0] and 0 <= y < mu.shape[1]:
                    P[x,y] += mu[x,y]
                
    return mu * np.exp(-P), P