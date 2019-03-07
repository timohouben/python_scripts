#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import numpy as np
import ternary

xyz = np.loadtxt('data.txt', delimiter=',')
nbins = 11

H, b = np.histogramdd((xyz[:, 0], xyz[:, 1], xyz[:, 2]),
                      bins=(nbins, nbins, nbins), range=((0, 1), (0, 1), (0, 1)))
H = H / np.sum(H)

# 3D smoothing and interpolation
from scipy.ndimage.filters import gaussian_filter
kde = gaussian_filter(H, sigma=2)
interp_dict = dict()
binx = np.linspace(0, 1, nbins)
for i, x in enumerate(binx):
    for j, y in enumerate(binx):
        for k, z in enumerate(binx):
            interp_dict[(i/10, j/10, k/10)] = kde[i, j, k]

print(interp_dict)

fig, tax = ternary.figure(scale=1)
tax.heatmap(interp_dict)
tax.show()
