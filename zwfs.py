import numpy as np


def simple_reconstruction_phase(Ic):
    """Performs a simple reconstruction of the phase
    from a ZWFS image Ic

    N’Diaye et al. “Calibration of quasi-static aberrations in exoplanet direct-imaging 
    instruments with a Zernike phase-mask sensor”, Astronomy & Astrophysics 555 (2013)

    Parameters
    ----------
    Ic : numpy.ndarray or HCIPy Field
        ZWFS intensity image

    Returns
    -------
    numpy.ndarray or HCIPy Field
        phase estimation
    """
    return -1 + np.sqrt(2*Ic)

def rayleigh_range(wavelength, w0):
    return np.pi * w0**2 / wavelength

def make_gaussian_qinv(wavelength, w0, z=0):
    zr = rayleigh_range(wavelength, w0)
    q = z + 1j*zr
    return 1/q

def thin_lens(f):

    lens = np.array([[1,0,0,0],
                     [0,1,0,0],
                     [-1/f,0,1,0],
                     [0,-1/f,0,1]])
    
    return lens

def free_space(d):

    dist = np.array([[1,0,d,0],
                     [0,1,0,d],
                     [0,0,1,0],
                     [0,0,0,1]])
    
    return dist

def gaussian_prop(qinv, abcd):

    Qinv = np.array([[qinv,0],
                     [0,qinv]])
    
    A = abcd[0:2, 0:2]
    B = abcd[0:2, 2:]
    C = abcd[2:, 0:2]
    D = abcd[2:, 2:]
    
    num = C + D @ Qinv
    den = A + B @ Qinv
    Qpinv = num @ np.linalg.inv(den)

    return Qpinv

def transversal_phase(Qpinv,r):
    """compute the transverse gaussian phase of a gaussian beam
    taken from poke.beamlets

    Parameters
    ----------
    Qpinv : numpy.ndarray
        N x 2 x 2 complex curvature matrix
    r : numpy.ndarray
        N x 2 radial coordinate vector

    Returns
    -------
    numpy.ndarray
        phase of the gaussian profile
    """

    transversal = (r[...,0]*Qpinv[...,0,0] + r[...,1]*Qpinv[...,1,0])*r[...,0]
    transversal = (transversal + (r[...,0]*Qpinv[...,0,1] + r[...,1]*Qpinv[...,1,1])*r[...,1])/2

    return transversal