#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import math
import ternary
from ternary.helpers import convert_coordinates_sequence
import random
# Make images higher resolution and set default size
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (4, 4)



scale = 1
qs = ((0.1,0.6,0.4),(2,2,0),(3,3,0),(4,4,0),(5,5,0),(6,6,0))
limits = {'b':(1,10),'l':(1,10),'r':(1,10)}
qs_plot = convert_coordinates_sequence(qs, scale, limits, axisorder='blr')
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 10)
tax.scatter(qs_plot, marker='s', color='red', label="Red Squares")
tax.show()
