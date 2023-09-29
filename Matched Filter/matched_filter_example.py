import numpy as np
import matched_filter_wrapper as mfw
import matplotlib.pyplot as plt

length = 1024
noise = 0.2 #the height of the peak is 6, so it should be easy to get high snr
width = 20.0 #for this example, I'm only checking one width, and all my sources are that exact width

def newsource (pos, freqs):
    return 6.0*np.exp(-0.5*((freqs-pos)/width)**2)

data = np.zeros(length)
frequencies = np.linspace(300,1500, length)

#placing some sources in the data
locations = [350, 600, 1000, 1301]
for i in range(len(locations)):
    data += newsource(locations[i], frequencies)
    
#generating noise
rng = np.random.default_rng()
data += rng.normal(0,noise,length)

#plt.plot(frequencies,data)
#plt.show()

peak_detections = mfw.matched_filter(data, frequencies, noise, 15, width, 8)
print("The peaks are in the ranges:")
print(peak_detections)
print("Which correspond to the frequency ranges:")
print(frequencies[peak_detections])
