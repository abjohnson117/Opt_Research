import numpy as np
import pywt
from scipy.ndimage import correlate, convolve
from scipy.fftpack import dct

def max_function(x, num_compare):
    return np.array([i if i >= num_compare else num_compare for i in x], dtype=float)

def prox_op(x,lambd):
    return np.sign(x)*max_function(np.abs(x)-lambd,0)

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

def blur_operator(org, reshape=True, shape=(9,9), sigma=1, mode="nearest"):
    if reshape:
        m = int(np.sqrt(org.shape[0]))
        org = np.reshape(org, (m,m))

    psf = fspecial(shape,sigma)
    blurred = correlate(org, psf, mode=mode)
    blurred += np.random.normal(0,1e-4,size=(m,m))
    blurred = blurred.T
    if reshape:
        blurred = blurred.flatten("F")
    
    return blurred

def dctshift(psf, center=(4,4)):
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

def evals_blur(psf, shape=(9,9), sigma=1):
    a1 = dctshift(psf)
    


def grad(x):
    return 2*(ATA@x)