# ** oceandisp **

Python code for computing the normalized dispersion relation of linear damped waves on the equatorial beta plane. This code computes the dispersion relation for Rayleigh friction and Laplacian friction numerically. 

___

## Additional Python libraries required:
> numpy, matplotlib, cmath



## Sample Code:
```
import numpy as np
import oceandisp as od


sigma = np.arange(0.001, 3, 0.001)
k = od.eqdisp(sigma, 0.1, mode=1, disp='rayleigh', plot=False)
```

## Test code:

```
import oceandisp as od
od.example(disp='rayleigh')
```

## Reference
Ref: P. Amol, D.Shankar (2024), Dispersion diagrams of linear damped waves on the equatorial beta plane, Ocean Modelling

## Contact
Contact prakashamol@gmail.com for any further information.
 
## License
The code is licensed under the MIT license.



