import numpy as np
import pywt
from scipy.ndimage import correlate

def max_function(x, num_compare):
    return np.array([i if i >= num_compare else num_compare for i in x], dtype=float)

def prox_op(x,lambd):
    return np.sign(x)*max_function(np.abs(x)-lambd,0)

def wavelet_operator_1d(org):
    coeffs = pywt.wavedec(org, wavelet="haar", level=2)
    wav_x, keep = pywt.coeffs_to_array(coeffs)
    return (wav_x,keep)


def blur_operator(org, reshape=True, shape=(3,3), sigma=0.5, mode="nearest"):
    if reshape:
        org = np.reshape(org, (int(np.sqrt(org.shape[0])),int(np.sqrt(org.shape[0]))))
    
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

    psf = fspecial(shape,sigma)
    blurred = correlate(org, psf, mode=mode)
    return blurred.T

def grad(x):
    return 2*(ATA@x)