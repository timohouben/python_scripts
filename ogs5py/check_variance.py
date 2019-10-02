import numpy as np

# EIN BEISPIEL FÜR LENNART

# Autocompletion: LENNART
# BEISPIEL


arimean = []
harmean = []
geomean = []

realizations = 10

for i in range(realizations):
    #np.random.seed(1337)
    mean=-10
    sigma=1
    size=100
    kf_list = np.random.lognormal(mean=mean,sigma=sigma,size=i*size)

    from scipy.stats.mstats import gmean, hmean

    arimean.append(np.mean(kf_list))
    harmean.append(hmean(kf_list))
    geomean.append(gmean(kf_list))


import matplotlib.pyplot as plt

x = [i*size for i in range(realizations)]

#plt.plot(x, arimean, label="arimean")
#plt.plot(x, harmean, label="harmean")
#plt.plot(x, geomean, label="geomean")
plt.semilogy(x, arimean, label="arimean")
plt.semilogy(x, harmean, label="harmean")
plt.semilogy(x, geomean, label="geomean")
plt.semilogy(sorted(kf_list), label="kf values")
#plt.ylim(0.00004,0.00005)
plt.ylabel("mean of samples")
plt.xlabel("number of samples")
plt.legend()
plt.show()

plt.hist(kf_list, bins=100)
max(kf_list)
min(kf_list)
