import numpy as np
import math as math
import matplotlib.pyplot as plt



''' Solution for the GWHead vs. distance (Dupuit 1863, from Chesnaux 2005) Depuit Forchenheimer Ellipse
    h = GWhead
    w = recharge
    k = sat. hydraulic conductivity (LT-1)
'''

w = 0.000000006              #Neubildung: 0,000000006 = entspricht etwa 200 mm/a
k = 0.000005                 #hydr. Leitfaehigkeit
l = 1000                     #Laenge GWScheide bis Vorfluter
hl = 29                      #Pegelstand Vorfluter
x = np.linspace(0, l, 100)   #moegliche x-Werte
head = []

def gwhead(w,k,l,hl,x):
    h = math.sqrt( w / k * ( l**2 - x**2 ) + hl**2 )
    return h

for i in x:
     h = gwhead(w,k,l,hl,i)
     head.append(h)
#   h[i, 1] = math.sqrt( w / k * ( l**2 - x[i]**2 ) + hl**2 )



#for i in x:
#   h[i] = gwhead(w,k,l,hl,x)

plt.plot(x,head, label="gwhead")
plt.title("gwhead")
plt.xlabel("distance")
plt.ylabel("gwhead")
plt.show()
 
