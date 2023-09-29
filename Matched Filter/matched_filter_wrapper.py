import numpy as np
import ctypes
import math
import matplotlib.pyplot as plt
from numpy.ctypeslib import ndpointer
import os
import sys


module_fn = 'mfs.so'
if sys.platform.startswith('darwin'):
    # macos -- link against mfs-mac.so
    module_fn = 'mfs-mac.so'

mfLib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), module_fn))
cpp_matched_filter = mfLib.matched_filter
cpp_matched_filter.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_uint, ctypes.c_double, ctypes.c_double, ctypes.c_double, ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS"), ctypes.c_uint, ctypes.POINTER(ctypes.c_int),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS")]

def matched_filter(data_input, frequency_input, data_noise, signal_threshold, template_width, max_detections, plot_output=False):
    
    #checking that all the inputs are correct
    if (data_input.ndim > 1):
        raise ValueError ("data_input must be one-dimensional")
    if (frequency_input.ndim > 1):
        raise ValueError ("frequency_input must be one-dimensional")
    if not math.log2(data_input.shape[0]).is_integer():
        raise ValueError ("input length must be power of 2")
    if frequency_input.shape[0] != data_input.shape[0]:
        raise ValueError ("Lengths of frequency and data arrays do not match")  
    #the other three inputs just have to be positive, so it should be fine to do this.
    assert(data_noise > 0); assert(signal_threshold > 0); assert(template_width > 0);
    
    rescaled_width = template_width/(frequency_input[-1] - frequency_input[0])*frequency_input.shape[0]
    
    detections = np.empty(2*max_detections,dtype=np.uint32)
    n_detected = ctypes.c_int();
    mfop = np.empty(2*data_input.shape[0])
    cpp_matched_filter(data_input, data_input.shape[0], data_noise, signal_threshold, rescaled_width, detections, max_detections, ctypes.byref(n_detected), mfop)
    if plot_output:
        plt.plot(mfop)
        plt.show()
    
    return detections[:2*n_detected.value].reshape([n_detected.value, 2])
