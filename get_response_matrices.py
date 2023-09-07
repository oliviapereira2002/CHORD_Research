from channelization_functions import get_response_matrix
import numpy as np

'''Example script for generating response matrices'''

# select desired upchannelization factors
U = [1, 2, 4, 8, 16, 32, 64]

# load in file containing frequencies
freqs = np.load('re-sampled_frequencies.npy')

# iterate through desired upchannelization factors to generate matrices
for u in U:

    # calling function the get arrays
    R, chans, norm = get_response_matrix(freqs, u)

    # saving as .npy files to be opened for later upchannelization
    np.save('R_' + str(u) + '.npy', R)
    np.save('chans_' + str(u) + '.npy', chans)
    np.save('norm_' + str(u) + '.npy', norm)