"""
	Functions to compute normalized dispersion relation of linear damped waves 
	on the equatorial beta plane. The dispersion relation can be computed numerically 
	or analytically for Rayleigh friction and numerically for Laplacian friction.


	Use eqdisp(sigma, gamma, mode, disp, plot) for general purpose. See details below:

        	Input for eqdisp 
        	----------------
        	sigma: array containing frequencies
        	gamma: friction coeff.
        	mode: meridional mode number
		disp: 'rayleigh' or 'laplace'
		plot: Boolean (True/False)
		

		Sample code:	
		------------
		import numpy as np
		import oceandisp as od

		sigma = np.arange(0.001, 3, 0.001)
		k = od.eqdisp(sigma, 0.1, mode=1, disp='rayleigh', plot=False)

		Sample example code for plotting:
		--------------------

		od.example(disp='rayleigh')

	List of functions: 
		1. raydispa() = Analytically compute dispersion curve with Rayleigh friction
		2. raydispn() = Numerically compute dispersion curve with Rayleigh friction
		3. lapdisp() = Numerically compute dispersion curve with Laplacian friction
		4. eqdisp()  = General function to numerically compute both Rayleigh and Laplacian friction	
		5. example() = To test the code

	Ref: P. Amol, D.Shankar (2024), Dispersion diagrams of linear damped waves 
	     on the equatorial beta plane, Ocean Modelling

	version 0.1	
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import cmath


def raydispa(sig, gamma, mode=1):
    """
        Function to create normalised equatorial dispersion relation for a viscid solution.
	The solutions are obtained analytically for k(sigma).

        Input
        -----
        sig: array containing frequency range
        gamma: friction coeff.
        mode: meridional mode number


	Output
	------
	Output is an array containing wavenumbers (k).  
	The solution has two roots in complex form.

	Sample code:
	------------
	import numpy as np
	import dispersion as dp

	sigma = np.arange(0.001, 3, 0.001)
	k = dp.raydispa(sigma, 0.1)

    """


    beta = 1
    c = 1	

    # Kelvin wave dispersion    
    if mode == -1:
        real1 = sig**2 - gamma**2
        imag1 = -2*sig*gamma


        mag = (real1**2 + imag1**2)**0.5


        kreal1 = ((mag + real1)/2)**0.5
        kreal2 = -1*kreal1
        kimag1 = ((mag - real1)/2)**0.5
        kimag2 = -1*kimag1

        root = np.array([kreal1 + 1j*kimag1, kreal2 + 1j*kimag2])
        root = root.T
        return root 


    # Rossby and Yanai wave dispersion
    w_fr = -beta*sig/(2*(sig**2 + gamma**2))
    w_fr_im = beta*gamma/(2*(sig**2 + gamma**2))


    kreal1 = np.zeros(len(sig))
    kreal2 = np.zeros(len(sig))
    kimag1 = np.zeros(len(sig))
    kimag2 = np.zeros(len(sig))


    if gamma == 0: 	#Inviscid case
        for i, sig1 in enumerate(sig):
            root_term = beta**2/(4*sig1**2) + sig1**2/c**2 - (2*mode+1)
            k1 = -beta/(2*sig1) + cmath.sqrt(beta**2/(4*sig1**2) + sig1**2/c**2 - (2*mode+1))
            k2 = -beta/(2*sig1) - cmath.sqrt(beta**2/(4*sig1**2) + sig1**2/c**2 - (2*mode+1))
            kreal1[i] = k1.real
            kreal2[i] = k2.real
            kimag1[i] = k1.imag
            kimag2[i] = k2.imag
    else:
        rossby_term = w_fr**2
        gravity_term = sig**2/c**2
        mode_term = -(2*mode + 1)*beta/c

        rossby_re_fr = -1*w_fr_im**2
        gravity_re_fr = -1*gamma**2/c**2

        rossby_im_fr = -1*(gamma*sig*beta**2)/(2*(sig**2 + gamma**2)**2)
        gravity_im_fr = 2*gamma*sig/c**2

        wave_term = rossby_term + gravity_term + mode_term

        real_term = wave_term + rossby_re_fr + gravity_re_fr
        imag_term = rossby_im_fr + gravity_im_fr

        sgn = np.sign(imag_term, out=np.ones_like(imag_term), where=imag_term!=0)

        mag = (real_term**2 + imag_term**2)**0.5

        kreal1 = w_fr + ((mag + real_term)/2)**0.5
        kreal2 = w_fr - ((mag + real_term)/2)**0.5

        kimag1 = w_fr_im + sgn*((mag - real_term)/2)**0.5
        kimag2 = w_fr_im - sgn*((mag - real_term)/2)**0.5

        root = np.array([kreal1 + 1j*kimag1, kreal2 + 1j*kimag2])
        root = root.T

    return root 



def raydispn(sigma, gamma, mode=1, method='poly'):
    """
        Function to create normalised equatorial dispersion with Rayleigh friction.
	The solutions are obtained numerically for k(sigma).

	For Kelvin modes (l=-1), roots are calculated analytically.


        Input
        -----
        sigma: array containing frequency range
        gamma: friction coeff.
        mode: meridional mode number


	Output
	------
	Output is an array containing wavenumbers (k).  
	The solution has two roots in complex form.

	Sample code:
	------------
	import numpy as np
	import dispersion as dp

	sigma = np.arange(0.001, 3, 0.001)
	k = dp.raydispn(sigma, 0.1)

    """


    beta = 1
    c = 1
    emode = 2*mode+1
    sigma = np.array(sigma)
    root = np.empty([sigma.size, 2], dtype=complex)	

    if mode == -1:
        root = raydispa(sigma, gamma, mode=mode)
        return root

    for i, sig in enumerate(sigma):
            a2 = c**2*sig + 1j*c**2*gamma
            a1 = beta*c**2
            ac = -1*sig**3 + emode*beta*c*sig + 3*gamma**2*sig - 3j*sig**2*gamma + 1j*gamma**3 + 1j*gamma*emode*beta*c
            ac = -1*(sig + 1j*gamma)**3 + emode*(sig + 1j*gamma)*beta*c
            p = [a2, a1, ac]

            if method == 'roots':
                    root[i, :] = np.roots(p)
            elif method == 'poly':
                    root[i, :] = poly.polyroots(p[::-1])
            else:
                    raise ValueError("Method option must be roots or poly.")

    return root


def lapkeldisp(sigma, gamma):
	kel1 = (-1j + 1j*np.sqrt(1 - 4j*sigma*gamma))/(2*gamma)
	kel2 = (-1j - 1j*np.sqrt(1 - 4j*sigma*gamma))/(2*gamma)
	kel3 = (1j + 1j*np.sqrt(1 - 4j*sigma*gamma))/(2*gamma)
	kel4 = (1j - 1j*np.sqrt(1 - 4j*sigma*gamma))/(2*gamma)


	root = np.array([kel1, kel2, kel3, kel4])
	return root.T

def lapdisp(sigma, gamma, mode=1, method='poly'):
        """
           Function to create normalised equatorial dispersion relation with Laplacian friction.
	   The solutions are obtained numerically for k(sigma).

	   For Kelvin modes (l=-1), roots are calculated analytically.

           -----   
           Input
           -----
           sig: array containing frequency range (sigma)
           gamma: friction coeff.
           mode: meridional mode number

           -----
           Output
           -----
	   Output is an array containing wavenumbers (k).  
	   The solution has six roots in complex form.
	
	   Example:
	   root = lapdispnum([0.001, 0.01, 0.1], 0.1)

       """

        beta = 1
        c = 1
        sigma = np.array(sigma)

        if mode == -1:
                root = lapkeldisp(sigma, gamma)
                return root

        emode = 2*mode+1
        root = np.empty([sigma.size, 6], dtype=complex)
        for i, sig in enumerate(sigma):
		# Calculate the 
                a6 = 1j*gamma**3
                a5 = 0
                a4 = 3*sig*gamma**2 + 1j*gamma*c**2
                a3 = 0
                a2 = c**2*sig - 3j*sig**2*gamma + 1j*gamma*emode*beta*c
                a1 = beta*c**2
                ac = -1*sig**3 + emode*beta*c*sig
                p = [a6, a5, a4, a3, a2, a1, ac]
                if method == 'roots':
                        root[i, :] = np.roots(p)
                elif method == 'poly':
                        root[i, :] = poly.polyroots(p[::-1])
                else:
                        raise ValueError("Method option must be roots or poly.")

        return root


def plotdisp(sigma, k, disp, gamma, mode):
	params = {'legend.fontsize': 'x-large',
	          'figure.figsize': (8, 4),
        	  'axes.labelsize': 'x-large',
        	  'axes.titlesize':'x-large',
        	  'xtick.labelsize':'x-small',
        	  'ytick.labelsize':'x-small'}

	plt.rcParams.update(params)
	fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

	ax[0].axvline(0, color='grey')
	ax[0].axhline(0, color='grey')
	ax[1].axvline(0, color='grey')
	ax[1].axhline(0, color='grey')
	
	for i in range(k.shape[1]):
#		ax[0].plot(k[:, i].real, sigma, color='k')
#		ax[1].plot(k[:, i].imag, sigma, color='r')
		ax[0].scatter(k[:, i].real, sigma, color='k', s=2, edgecolors='none')
		ax[1].scatter(k[:, i].imag, sigma, color='r', s=2, edgecolors='none')

	ax[0].set_title('Real')
	ax[1].set_title('Imaginary')

	ax[0].text(0.05, 0.9, '$\gamma$={}'.format(gamma), transform=ax[0].transAxes, fontsize=8, bbox={'facecolor':'white', 'ec':'white', 'alpha':1})
	ax[1].text(0.05, 0.9, '$\gamma$={}'.format(gamma), transform=ax[1].transAxes, fontsize=8, bbox={'facecolor':'white', 'ec':'white', 'alpha':1})
	ax[0].text(0.05, 0.85, '$\ell$={}'.format(mode), transform=ax[0].transAxes, fontsize=8, bbox={'facecolor':'white', 'ec':'white', 'alpha':1})
	ax[1].text(0.05, 0.85, '$\ell$={}'.format(mode), transform=ax[1].transAxes, fontsize=8, bbox={'facecolor':'white', 'ec':'white', 'alpha':1})
	
	fig.text(0.5, 0.03, 'Wave number $k\sqrt{c/(\\beta)}$', ha='center', va='center')
	fig.text(0.07, 0.5, 'Frequency $\sigma/\sqrt{(\\beta c)}$', ha='center', va='center', rotation='vertical')
	plt.suptitle(f'Dispersion relation [k($\sigma$)] - {disp.capitalize()} friction')
	plt.xlim([-11, 11])
	plt.show()



def eqdisp(sigma, gamma, mode=1, disp='rayleigh', plot=False):

        method = 'poly'
	# Check for errors in input
        if not isinstance(sigma, (list, tuple, np.ndarray)):
                raise TypeError("sigma must be a list or an array")
        if not isinstance(mode, int):
                raise TypeError("mode must be an integer")
        if mode < -1:
                raise ValueError("mode must be greater than or equal to -1")
        if gamma < 0:
                raise ValueError("gamma must be greater than or equal to 0")
        if not isinstance(plot, bool):
                raise TypeError("plot must be boolean type (True or False)")

        if disp == 'rayleigh' or gamma == 0:
        #       root = raydispn(sigma, gamma, mode, method)
                root = raydispa(sigma, gamma, mode)
        elif disp == 'laplace':
                root = lapdisp(sigma, gamma, mode, method)
        else:
                raise ValueError("disp option must be rayleigh or laplace.")	


        if plot:
               plotdisp(sigma, root, disp, gamma, mode)

        return root


def example(disp='rayleigh'):

	sigma = np.arange(0.001, 3, 0.001)
	if disp=='rayleigh':
		gamma = 0.1
	else:
		gamma=0.01

	eqdisp(sigma, gamma, mode=1, disp=disp, plot=True)
		

if __name__ == '__main__':
        example()
