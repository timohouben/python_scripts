from transfer_functions import discharge_ftf
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt

# time ste size
time_step_size = 86400
# amount of data points
length = 5000
# according frequencies
f = (abs(fftpack.fftfreq(length, time_step_size))[:int(round(length / 2))])[1:]
d_list = [10, 7.5, 5, 2.5, 1, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.001, 0.0075, 0.005, 0.0025, 0.0001]
aquifer_length_list = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 5000, 10000, 25000, 50000, 75000, 10000]

plt.figure(figsize=(16,9))

for d, color in zip(d_list, range(len(d_list))):
    power_spectrum = discharge_ftf(f, d, aquifer_length_list[4])
    plt.loglog(power_spectrum, label="D = " + str(d), color=plt.cm.Reds(color*25 + 100), linestyle="--")
    plt.ylim(1e-5, 2)
    #plt.xlim(1e-5, 2)

for aquifer_length, color in zip(aquifer_length_list,range(len(aquifer_length_list))):
    power_spectrum = discharge_ftf(f, d_list[6], aquifer_length)
    plt.loglog(power_spectrum, label="L = " + str(aquifer_length), color=plt.cm.Blues(color*25))

plt.ylabel("Power Spectrum")
plt.xlabel("Frequency")
plt.legend(ncol=2)
plt.title("Tranfer Function for the Baseflow with different aquifer lengths [m] and diffusivities [m2/s]\n D = 0.1, L = 1000 ")
plt.show()
