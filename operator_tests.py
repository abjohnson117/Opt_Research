import numpy as np
from scipy.ndimage import correlate
from operators import prox_op, wavelet_operator_1d, blur_operator, blur_adjoint

def test_prox():
    return 0

def test_wavelet():
    return 0

def test_grad():
    return 0

def test_blur(x,b):
    if len(x.shape) > 1:
        x = x.flatten("F")
    if len(b.shape) > 1:
        b = b.flatten("F")
    error = np.dot(blur_operator(x),b) - np.dot(x, blur_adjoint(b))
    if error < 1e-4:
        return True
    else:
        return False