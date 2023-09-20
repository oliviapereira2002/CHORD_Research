import unittest
import numpy as np
import pytest
import channelization_functions

# testing functions used for response matrix
c = np.reshape(np.arange(10), (1, len(np.arange(10))))
f = np.reshape(np.linspace(0, 10, 50), (50, 1))
M = 4
N = 4096
U = 2
j = np.reshape(np.arange(M*N), (1, M*N))
k = np.reshape(np.arange(M*U), (1, M*U))
submtx_upchan = np.tile(np.arange(U), [f.shape[0], 1]).T
submtx_upchan = (U-1) / U - 2*submtx_upchan / U + 2*f[:,0]

def test_window():
    assert type(channelization_functions.window(j, M, N)) == np.ndarray
    assert np.shape(channelization_functions.window(j, M, N)) == np.shape(j)

def test_exponential_chan():
    assert type(channelization_functions.exponential_chan(j, f, N)) == np.ndarray
    assert (np.shape(channelization_functions.exponential_chan(j, f, N)) 
                == (np.shape(f)[0], np.shape(j)[0], np.shape(j)[1]))
def test_weight_chan():
    assert type(channelization_functions.weight_chan(f, M, N)) == np.ndarray
    assert np.shape(channelization_functions.weight_chan(f, M, N) == (np.shape(f)[0], 1))

def test_exponential_upchan():
    assert type(channelization_functions.exponential_upchan(submtx_upchan, k)) == np.ndarray
    assert np.shape(channelization_functions.exponential_upchan(submtx_upchan, k)) == (U, np.shape(f)[0], M*U)

def test_response_matrix():
    assert (np.shape(channelization_functions.response_mtx(c, f, M, N, U))
                == (np.shape(c)[1] * U, len(f)))

def test_freq_unit_strip():
    frequency = 300 # making sure it sends the lowest CHORD frequency to 0 in index space
    assert channelization_functions.freq_unit_strip(frequency) == 0

def test_freq_unit_add():
    index = 0
    assert channelization_functions.freq_unit_add(index) == 300

def test_get_chans():
    assert len(channelization_functions.get_chans(1398, 1398)) == 2
    assert len(channelization_functions.get_chans(1399, 1398)) == 0


