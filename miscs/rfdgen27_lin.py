    import numpy as np

import matplotlib.pyplot as plt



startx=0
endx=720000
incx=7200
starty=1
endy=2


a = np.arange(startx, endx+incx, incx)
b = np.linspace(starty, endy, len(a))
c = np.vstack((a,b)).T
np.savetxt("out.txt", c)


plt.plot(b,a, label="your .rfd")
plt.title("Your .rfd")
plt.xlabel("Zeit")
plt.ylabel("Wert")
plt.show()
 