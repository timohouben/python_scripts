from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def function_CP(x):
    #y = (1 - ( 1 / (x) ))
    #y = 1 / ((1 + x**0.2)**2)
    #y = 50000 + ( 1 / (1 + x) )
    #y = 60000*10**(-1.5*x)
    y = 56347 * x**2 - 105913 * x + 49841
    return y

def function_K(x):
    y = x**2 * ( 1 - ( 1 - x))
    return y

startx = 0.001
endx = 1


a = np.linspace(startx, endx, 200)          #lineare Aufteilung
         

c = np.vstack((a, function_CP(a))).T         #Transponieren
d = np.vstack((a, function_K(a))).T
np.savetxt("function_CP.rfd", c)
np.savetxt("function_K.rfd", d)
print c
print d

plt.figure(1)
plt.subplot(211)
plt.plot(a, function_CP(a))
plt.ylabel("CP")
plt.subplot(212)
plt.plot(a, np.gradient(function_CP(a)))
plt.ylabel("gradient CP")
plt.xlabel("Saturation")

plt.figure(2)
plt.subplot(211)
plt.plot(a, function_K(a))
plt.ylabel("Krel")
plt.subplot(212)
plt.plot(a, np.gradient(function_K(a)))
plt.ylabel("gradient Krel")
plt.xlabel("Saturation")

plt.show()