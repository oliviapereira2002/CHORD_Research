import astropy.units as u
import numpy as np
from unit_converter import GalaxyProfile # to convert profiles 
from channelization_functions import channelize_catalogue, channelize_map

'''channelizing a catalogue'''
U = 16  # choose an upchannelization factor
catalogue_filepath = '/Users/oliviapereira/Desktop/channelization_project/source_files/HI_Catalog.txt'
R_filepath = '/Users/oliviapereira/Desktop/channelization_project/matrices/1400/R_' + str(U) + '.npy'
norm_filepath = '/Users/oliviapereira/Desktop/channelization_project/matrices/1400/norm_' + str(U) + '.npy'

channelized_catalogue = channelize_catalogue(U, catalogue_filepath, R_filepath, norm_filepath)

'''channelizing a map'''
U = 16  # choose an upchannelization factor
map_filepath = '/Users/oliviapereira/Desktop/channelization_project/source_files/diffuse_emission.h5'
R_filepath = '/Users/oliviapereira/Desktop/channelization_project/matrices/1400/R_' + str(U) + '.npy'
norm_filepath = '/Users/oliviapereira/Desktop/channelization_project/matrices/1400/norm_' + str(U) + '.npy'
chans_filepath = '/Users/oliviapereira/Desktop/channelization_project/matrices/chans_' + str(U) + '.npy'

# will save the map until a specified save_title
channelize_map(U, map_filepath, R_filepath, norm_filepath, chans_filepath, 'my_map_name.h5')


