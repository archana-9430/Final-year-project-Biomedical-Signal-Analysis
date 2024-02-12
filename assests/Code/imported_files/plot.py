import matplotlib.pyplot as plt
import sys

def on_press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'escape':
        plt.close()


def plot_signal(x : list ,y : list , type : str = 'b' , x_label = None , y_label = None , title = None):

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax.plot(x , y , type)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
