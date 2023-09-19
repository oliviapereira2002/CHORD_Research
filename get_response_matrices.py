from channelization_functions import get_response_matrix
import numpy as np

'''Example script for generating response matrices'''

# select desired upchannelization factors
U = [1, 2, 4, 8, 16, 32, 64]

# load in file containing frequencies
freqs = np.load('re-sampled_frequencies.npy')

# iterate through desired upchannelization factors to generate matrices
for u in U:

    observing_freqs = np.load('chans_' + str(U) + '.npy')
    # calling function the get arrays
    R, norm = get_response_matrix(freqs, observing_freqs, u)

    # saving as .npy files to be opened for later upchannelization
    np.save('R_' + str(u) + '.npy', R)
    np.save('norm_' + str(u) + '.npy', norm)