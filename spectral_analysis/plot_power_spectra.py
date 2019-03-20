# -*- coding: utf-8 -*
# ------------------------------------------------------------------------------
# python 2 and 3 compatible
from __future__ import division

# ------------------------------------------------------------------------------


def plot_spectrum(
    data,
    frequency,
    name=None,
    labels=None,
    path=None,
    lims=None,
    linestyle="-",
    marker="",
    grid="both",
    unit="[Hz]",
    heading="None",
    figtxt=None,
):
    """
    Function to plot one or multiple power spectra.

    Parameters
    ----------
    data : 2-D array
        Each row represents a seperate power spectrum.
    frequency : 1-D array
        Corresponding frequencies of data.
    name : string
        Name of file. If None, time is used.
    labels : X item list
        Labels for different power spectra as list in same order as data.
    path : string
        Path to store the image.
    lims : list with 2 tuples
        lims[0] = x limit as tuple (xmin,xmax)
        lims[1] = y limit as tuple (ymin,ymax)
        e.g. lims = [(1e-8,1e-4),(1e0,1e5)]
    linestyle : X item list
        List with linestyles for differenct spectra.
    marker : X item list
        List with marker for differenct spectra.
    grid : string
        "major", "minor", "both", "none"
    unit : string
        Unit of frequency.
    heading : string
        Provide a heading for the image. If None, no heading.
    figtxt : string (multiline possible)
        Provide an annotation for a box below the figure. If None, no annotaion.

    Yields
    ------
    One saved image in path.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    font = {"family": "Helvetica", "weight": "normal", "size": 20}
    plt.rc("font", **font)
    plt.rc("legend", fontsize=15)
    plt.figure(figsize=[20, 10], dpi=100)
    if np.ndim(data) == 1:
        plt.loglog(
            frequency,
            data,
            label=str(labels[0]),
            linewidth=1,
            linestyle=linestyle,
            marker=marker,
        )
    else:
        for i, spectrum in enumerate(data):
            print(np.shape(spectrum))
            plt.loglog(
                frequency,
                spectrum,
                label=labels[i],
                linewidth=1,
                linestyle=linestyle[i],
                marker=marker[i],
            )
    plt.grid(which=grid, color="grey", linestyle="-", linewidth=0.2)
    if lims != None:
        plt.xlim(lims[0])
        plt.ylim(lims[1])
    if heading != None:
        plt.title(heading)
    if labels != None:
        plt.ylabel("Spectral Density")
        plt.xlabel("Frequency %s" % unit)
    plt.legend(loc="best")
    if figtxt != None:
        plt.figtext(
            0.135,
            -0.05,
            figtxt,
            horizontalalignment="left",
            bbox=dict(boxstyle="square", facecolor="#F2F3F4", ec="1", pad=0.8, alpha=1),
        )
    if path != None:
        import datetime

        if name == None:
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        plt.savefig(path + "/" + name + ".png", pad_inches=0.5, bbox_inches="tight")
    plt.close()


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
    spectrum = abs(spectrum[: round(len(spectrum) / 2)]) ** 2
    # erwase first data point
    spectrum = spectrum[1:]
    print(len(spectrum))
    print(spectrum)
    # X contains the different locations
    X = np.linspace(0, aquifer_length - 1, int((aquifer_length / 10)))
    print(X)
    print(len(X))
    # Y contains the frequencies
    Y = abs(fftpack.fftfreq(len(input), time_step_size))[: round(len(input) / 2)][1:]
    print(len(Y))
    Z = np.zeros((len(Y), len(X)))
    for i, loc in enumerate(X):
        Z[:, i] = np.log10(
            shh_analytical(
                (Y, spectrum),
                Sy=0.00001,
                T=0.00001,
                x=loc,
                L=aquifer_length,
                m=5,
                n=5,
                norm=False,
            )
        )
    print(Z)
    print(len(Z))
    print(np.shape(Z))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(
        X, Y, Z, rstride=1, cstride=2, cmap=cm.jet, shade=False, linewidth=1
    )
    # surf = ax.plot_wireframe(X, Y, Z, rstride=0, cstride=5, cmap=cm.magma)
    # surf.set_edgecolors(surf.to_rgba(surf._A))
    surf.set_facecolors("white")
    # ax1 = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax.set_xlabel("location")
    ax.set_ylabel("frequency [Hz]")
    ax.set_zlabel("spectral density")
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
    spectrum = abs(spectrum[: round(len(spectrum) / 2)]) ** 2
    # erwase first data point
    spectrum = spectrum[1:]
    print(len(spectrum))
    print(spectrum)
    # X contains the different storativities
    # X = 10 ** np.linspace(1, 5, 5)
    X = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]
    # Y contains the frequencies
    Y = abs(fftpack.fftfreq(len(input), time_step_size))[: round(len(input) / 2)][1:]
    print(len(Y))
    Z = np.zeros((len(Y), len(X)))
    for i, S in enumerate(X):
        Z[:, i] = np.log10(
            shh_analytical(
                (Y, spectrum),
                Sy=S,
                T=0.05,
                x=500,
                L=aquifer_length,
                m=5,
                n=5,
                norm=False,
            )
        )
    print(Z)
    print(len(Z))
    print(np.shape(Z))
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = Axes3D(fig)
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, shade=False, linewidth=1)
    surf = ax.plot_wireframe(X, Y, Z, rstride=0, cstride=1, cmap=cm.magma)
    # surf.set_edgecolors(surf.to_rgba(surf._A))
    # surf.set_facecolors("white")
    # ax1 = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax.set_xlabel("storativity")
    ax.set_ylabel("frequency [Hz]")
    ax.set_zlabel("spectral density")
    plt.show()


if __name__ == "__main__":
    # plot_shh_anal_loc(aquifer_length=1000, time_step_size=86400)
    # plot_shh_anal_S(aquifer_length=1000, time_step_size=86400)

    # Test for function plot_spectrum
    from power_spectrum import power_spectrum
    import numpy as np

    frequency, data1 = power_spectrum(
        np.random.rand(1000), np.random.rand(1000), 86400, o_i="o"
    )
    frequency, data2 = power_spectrum(
        np.random.rand(1000), np.random.rand(1000), 86400, o_i="o"
    )
    frequency, data3 = power_spectrum(
        np.random.rand(1000), np.random.rand(1000), 86400, o_i="o"
    )
    data = np.vstack((data1, data2, data3))
    labels = ["head1", "head2", "head3"]
    linestyle = ["-", "--", ":"]
    path = "/Users/houben/Desktop/TEST"
    plot_spectrum(
        data,
        frequency,
        labels,
        path,
        lims=None,
        figtxt="nonoasndoand\nasdasd",
        linestyle=linestyle,
    )
