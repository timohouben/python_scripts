#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3

import math
import ternary
import numpy as np
import matplotlib

def tc_gelhar(T, L, S):
    return np.log10(L**2*S/3./T/86400)

def generate_heatmap_data(Tmin, Tmax, Smin, Smax, Lmin, Lmax, scale, log=True):
    data = dict()
    if log == True:
        Tmin, Tmax = np.log10(Tmin), np.log10(Tmax)
        Smin, Tmax = np.log10(Smin), np.log10(Smax)
    from matplotlib.mlab import frange
    print(Tmin,Tmax)
    for i, item in enumerate(frange(Tmin, Tmax, (Tmax-Tmin)/scale)):
        for j, jtem in enumerate(frange(Lmin, Lmax, (Lmax-Lmin)/scale)):
            for k, ktem in enumerate(frange(Smin, Smax, (Smax-Smin)/scale)):
                if log == True:
                    data[(i, j, k)] = tc_gelhar(10**(item), jtem, 10**(ktem))
                else:
                    data[(i, j, k)] = tc_gelhar(item, jtem, ktem)

    return data

Tmin, Tmax = 1e-5, 1e-2
Smin, Smax = 1e-5, 0.1
Lmin, Lmax = 10, 5000

limits = {'b': [Lmin, Lmax], 'l': [np.log10(Tmin), np.log10(Tmax)], 'r': [np.log10(Smin), np.log10(Smax)]}
scale = 10
data = generate_heatmap_data(Tmin, Tmax, Smin, Smax, Lmin, Lmax, scale)
#points = ternary.helpers.convert_coordinates_sequence(qs=data, scale=scale, li mits=limits, axisorder='blr')

# user defined colors and values
c1 = 1
c2 = 2
c3 = 3
c4 = 4
x1 = 0
x2 = c1/max(data.values())
x3 = c2/max(data.values())
x4 = c3/max(data.values())
x5 = c4/max(data.values())
x6 = 1
cdict = {
  'green': ((x1, 0, 1),
            (x2, 1, 1),
            (x3, 1, 0),
            (x4, 0, 0),
            (x5, 0, 0),
            (x6, 0, 0)),
  'blue':  ((x1, 0, 0),
            (x2, 0, 1),
            (x3, 1, 1),
            (x4, 1, 0),
            (x5, 0, 0),
            (x6, 0, 0)),
  'red' :  ((x1, 0, 0),
            (x2, 0, 0),
            (x3, 0, 0),
            (x4, 0, 1),
            (x5, 1, 1),
            (x6, 0, 0))
}

cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 2048)

#print(data)
print(len(data))

figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 8)
tax.heatmap(data,scale=scale, style="hexagonal", cmap=cmap,
cbarlabel='log(tc) [days]', vmax=None, vmin=0)
tax.set_title("charac. time, Gelhar 1974", pad=5)
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
# Draw Boundary and Gridlines
axes_colors = {'b': 'black', 'l': 'black', 'r':'black'}
tax.boundary(linewidth=1.5, axes_colors=axes_colors)
tax.gridlines(multiple=1, linewidth=2,
              horizontal_kwargs={'color':axes_colors['b']},
              left_kwargs={'color':axes_colors['l']},
              right_kwargs={'color':axes_colors['r']},
              alpha=0.2)
# Set Axis labels and Title
fontsize = 12
tax.left_axis_label("Transmissivity \nT [m2/s]", fontsize=fontsize, offset=0.2, color=axes_colors['l'])
tax.right_axis_label("Storativity \nS [-]", fontsize=fontsize, offset=0.2, color=axes_colors['r'])
tax.bottom_axis_label("Aq. Length \nL [m]", fontsize=fontsize, offset=0.2, color=axes_colors['b'])
# Set custom axis limits
tax.set_axis_limits(limits)
tax.get_ticks_from_axis_limits()
tax.set_custom_ticks(fontsize=10, offset=0.05, tick_formats='%.2f')
tax.show()

# Questions:
# Plot data correctly?
# offset of Title?
# Custom ticks for each axis? Rotation for bottom axis?
# Move color bar to the right?
# more ticks but less labeling
# color bar label
