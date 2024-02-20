import sys

import matplotlib.pyplot as plt
import numpy as np


def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()

    if event.key == 'escape':
        plt.close()

    if event.key == 'M' or 'm':
        plt.get_current_fig_manager().window.state('zoomed')


# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.subplots()

fig.canvas.mpl_connect('key_press_event', on_press)

ax.plot(np.random.rand(12), np.random.rand(12), 'go')
xl = ax.set_xlabel('easy come, easy go')
ax.set_title('Press a key')
plt.show()