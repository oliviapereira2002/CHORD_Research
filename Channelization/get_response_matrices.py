import channelization_functions as cf
import numpy as np

'''Example script for generating response matrices'''

# iterate through desired upchannelization factors to generate matrices
for U in [1, 2, 4, 8, 16, 32, 64]:
    fmin = 1404
    fmax = 1422

    # loading in channel locations
    channels = np.load('chans_' + str(U) + '.npy')

    # generating where the profiles should be re-sampled for the given U
    fine_freqs = cf.get_fine_freqs(channels) 

    # generating the response matrix
    R, channels, norm = cf.get_response_matrix(fine_freqs, U, fmin, fmax, viewmatrix = True)
    np.save('R_' + str(U) + '.npy', R)
    np.save('norm_' + str(U) + '.npy', norm)
