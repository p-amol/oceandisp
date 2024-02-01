import numpy as np
import oceandisp as od

sigma = np.arange(0.001, 3, 0.001)
k = od.eqdisp(sigma, 0.1, mode=1, disp='rayleigh', plot=True)
