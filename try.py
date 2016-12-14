import numpy as np
import time
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import pylab as plb

#   set up figure for plotting:
fig = plt.figure()
ax = fig.add_subplot(111)

#    plot limits
#ax.set_xlim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))
#ax.set_ylim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))

#    colors
colors = ['b', 'g', 'c']


for i_t in range(10000):
    x = range(i_t)
    y = np.square(x)

    ax.clear()
    plt.hold(True)
    #    plot limits
    #ax.set_xlim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))
    #ax.set_ylim(-(max(q0) + bodies[-1].L), +(max(q0) + bodies[-1].L))
    ax.plot(x, y)
    plt.pause(0.0001)
