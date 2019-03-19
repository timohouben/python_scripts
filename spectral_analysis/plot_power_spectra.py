#!/Library/Frameworks/Python.framework/Versions/3.6/bin/python3
# -*- coding: utf-8 -*


def plot_single_spectrum():
    """
    Function to plot a single power spectrum
    """

def plot_multiple_spectrum():
    """
    Function to plot multiple power spectra along e.g. along aquifer.
    """

def plot_shh_anal_loc(aquifer_length, time_step_size):
    """
    Function to plot multiple analytical power spectra along e.g. along aquifer.
    """

    import sys
    # add search path for own modules
    sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
    from shh_analytical import shh_analytical
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import scipy.fftpack as fftpack

    data_points = 100
    time_step_size = 86400
    aquifer_length = 1000
    # create an input signal
    np.random.seed(123456789)
    input = np.random.rand(data_points)
    spectrum = fftpack.fft(input)
    spectrum = abs(spectrum[:round(len(spectrum)/2)]) ** 2
    # erwase first data point
    spectrum = spectrum[1:]
    print(len(spectrum))
    print(spectrum)
    # X contains the different locations
    X = np.linspace(0, aquifer_length-1, int((aquifer_length/10)))
    print(X)
    print(len(X))
    # Y contains the frequencies
    Y = abs(fftpack.fftfreq(len(input), time_step_size))[:round(len(input) / 2)][1:]
    print(len(Y))
    Z = np.zeros((len(Y), len(X)))
    for i, loc in enumerate(X):
        Z[:,i] = np.log10(shh_analytical((Y, spectrum), Sy=0.00001, T=0.00001, x=loc, L=aquifer_length, m=5, n=5, norm=False))
    print(Z)
    print(len(Z))
    print(np.shape(Z))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=2, cmap=cm.jet, shade=False, linewidth=1)
    #surf = ax.plot_wireframe(X, Y, Z, rstride=0, cstride=5, cmap=cm.magma)
    #surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
    #ax1 = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax.set_xlabel('location')
    ax.set_ylabel('frequency [Hz]')
    ax.set_zlabel('spectral density')
    plt.show()

def plot_shh_anal_S(aquifer_length, time_step_size):
    """
    Function to plot multiple analytical power spectra along diff. storativities.
    """

    import sys
    # add search path for own modules
    sys.path.append("/Users/houben/PhD/python/scripts/spectral_analysis")
    from shh_analytical import shh_analytical
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    import scipy.fftpack as fftpack

    data_points = 5000
    time_step_size = 86400
    aquifer_length = 1000
    # create an input signal
    np.random.seed(123456789)
    input = np.random.rand(data_points)
    spectrum = fftpack.fft(input)
    spectrum = abs(spectrum[:round(len(spectrum)/2)]) ** 2
    # erwase first data point
    spectrum = spectrum[1:]
    print(len(spectrum))
    print(spectrum)
    # X contains the different storativities
    # X = 10 ** np.linspace(1, 5, 5)
    X = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    # Y contains the frequencies
    Y = abs(fftpack.fftfreq(len(input), time_step_size))[:round(len(input) / 2)][1:]
    print(len(Y))
    Z = np.zeros((len(Y), len(X)))
    for i, S in enumerate(X):
        Z[:,i] = np.log10(shh_analytical((Y, spectrum), Sy=S, T=0.05, x=500, L=aquifer_length, m=5, n=5, norm=False))
    print(Z)
    print(len(Z))
    print(np.shape(Z))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, shade=False, linewidth=1)
    surf = ax.plot_wireframe(X, Y, Z, rstride=0, cstride=1, cmap=cm.magma)
    #surf.set_edgecolors(surf.to_rgba(surf._A))
    #surf.set_facecolors("white")
    #ax1 = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax.set_xlabel('storativity')
    ax.set_ylabel('frequency [Hz]')
    ax.set_zlabel('spectral density')
    plt.show()


if __name__ == "__main__":
    #plot_shh_anal_loc(aquifer_length=1000, time_step_size=86400)
    plot_shh_anal_S(aquifer_length=1000, time_step_size=86400)
