import sys
import matplotlib.pyplot as plt


def on_press(event):
    # print('press', event.key)
    if event.key == 'escape':
        plt.close()

    elif event.key == 'M' or 'm':
        # print("Zoom!!")
        plt.get_current_fig_manager().window.state('zoomed')



def plot_signal_interactive(x : list ,y : list , style : str = 'b' , x_label = None , y_label = None , title = None):

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax.plot(x , y , style)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
