import numpy as np
from scipy import integrate

# helper function
def draw_sample(bins, heights, n):
    # cumulatively integrating 
    y_cumulative = integrate.cumtrapz(heights, initial = 0)
    draws = []

    # randomly sampling based on original probability
    for i in range(n):
        choice = np.random.uniform(low = 0, high = 1)
        draw = np.interp(choice, y_cumulative, bins)
        draws.append(draw)

    return np.array(draws)


# wrapper function
def draw_random_errors(phase_bins, phase_heights, amp_bins, amp_heights, output_shape):

    # normalizing the input heights to make it into a probability distribution
    phase_heights = phase_heights / integrate.trapz(np.abs(phase_heights))
    amp_heights = amp_heights / integrate.trapz(np.abs(amp_heights))
    
    # getting total random errors needed to populate the result matrix
    n = np.prod(output_shape)

    # getting phase errors for each matrix
    phase_draws = draw_sample(phase_bins, phase_heights, n).reshape(output_shape)
    amp_draws = draw_sample(amp_bins, amp_heights, n).reshape(output_shape)
    
    return phase_draws, amp_draws


# wrapper function for once we have the file 
def get_calibration_errors(output_shape, phase_file, amp_file):
    
    phase_bins, phase_heights = np.load(phase_file).T
    amp_bins, amp_heights = np.load(amp_file).T 
    phase_draws, amp_draws = draw_random_errors(phase_bins, phase_heights, amp_bins, amp_heights, output_shape)
    
    return phase_draws, amp_draws

# ***************************************

if (__name__ == "__main__"):
    ''' Example for doing a 1-D draw from our histograms '''
    import matplotlib.pyplot as plt

    ndraws = 100000
    nbins = 100 # for plotting results, has no effect on function

    phase_draws, amp_draws = get_calibration_errors((1, 1, ndraws), "visibility_phase_errors.npy", "visibility_amplitude_errors.npy")
    
    # re-loading in just to plot original distributio
    phase_bins, phase_heights = np.load("visibility_phase_errors.npy").T
    amp_bins, amp_heights = np.load("visibility_amplitude_errors.npy").T 

    # plotting the random draws
    plt.subplot(1,2,1)
    counts, bins = np.histogram(phase_draws, bins = nbins)
    plt.plot(phase_bins, phase_heights / np.max(phase_heights), label = 'original histogram', color = 'black', linewidth = 0.8)
    plt.stairs(counts / np.max(counts), edges = bins, label = 'phase draws,\nn = ' + str(ndraws), color = 'blue', alpha = 0.8)
    plt.title('Scaled Results of Random Draws from Phase Error Hist.')
    plt.xlabel('Phase error [rad]')
    plt.legend()

    plt.subplot(1,2,2)
    counts, bins = np.histogram(amp_draws, bins = nbins)
    plt.plot(amp_bins, amp_heights / np.max(amp_heights), label = 'original histogram', color = 'black', linewidth = 0.8)
    plt.stairs(counts / np.max(counts), edges = bins, label = 'amplitude draws,\nn = ' + str(ndraws), color = 'red', alpha = 0.8)
    plt.title('Scaled Results of Random Draws from Amplitude Error Hist.')
    plt.xlabel('Amplitude error [%]')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
