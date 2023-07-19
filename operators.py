import numpy as np
import pywt
from scipy.ndimage import correlate, convolve
from scipy.signal import convolve2d
from scipy.fftpack import dct

def max_function(x, num_compare):
    return np.array([i if i >= num_compare else num_compare for i in x], dtype=float)

def prox_op(x,lambd):
    return np.sign(x)*max_function(np.abs(x)-lambd,0)

def wavelet_operator(orig):
    coeffs = pywt.wavedec2(orig, wavelet="haar", level=3)
    wav_x, _ = pywt.coeffs_to_array(coeffs)
    wav_x = wav_x[0:m,0:m]
    W = np.reshape(wav_x, (m**2,1))
    return W

def wavelet_operator_1d(org):
    coeffs = pywt.wavedec(org, wavelet="haar", level=2)
    wav_x, keep = pywt.coeffs_to_array(coeffs)
    return (wav_x,keep)

def fspecial(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def blur_operator(org, reshape=True, shape=(9,9), sigma=1, mode="reflect"):
    if reshape:
        m = int(np.sqrt(org.shape[0]))
        org = np.reshape(org, (m,m))

    psf = fspecial(shape,sigma)
    blurred = correlate(org, psf, mode=mode)
    # blurred = convolve2d(org, psf, mode="same", boundary=boundary)
    # blurred += np.random.normal(0,1e-4,size=(m,m)) #TODO: Add this separately
    blurred = blurred.T
    if reshape:
        blurred = blurred.flatten("F")
    
    return blurred

def dctshift(psf, center=(4,4)):
    """Taken from Deblurring Images to compute first column of A  matrix"""
    m, n = psf.shape[0], psf.shape[1]
    i = center[0]
    j = center[1]
    k = min([i,m-i,j,n-j])

    PP = psf[i-(k):i+(k+1),j-(k):j+(k+1)]
    Z1 = np.diag(np.ones(k+1), k)
    Z2 = np.diag(np.ones(k), k+1)

    PP = Z1@PP@Z1.T + Z1@PP@Z2.T + Z2@PP@Z1.T + Z2@PP@Z2.T 
    Ps = np.zeros((m,n))
    Ps[0:2*k+1,0:2*k+1] = PP

    return Ps

def evals_blur(psf):
    """Calculates eigenvalues according to equation in Deblurring Images"""
    a1 = dctshift(psf)
    
    e1 = np.zeros_like(a1)
    e1[0,0] = 1

    S = dct(dct(a1, axis=0), axis=1) / dct(dct(e1,axis=0), axis=1)
    #TODO: CHECK THIS
    return np.max(S), S


def grad(x,b):
    return 