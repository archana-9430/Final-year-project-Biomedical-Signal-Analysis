'''
Provides customised interactive matplotlib functionality for annotation
'''
import matplotlib.pyplot as plt

def on_press(event):
    '''
    Callback function to handle keypress events
    '''
    # print('press', event.key)
    if event.key == 'escape':
        plt.close()

    elif event.key == 'm':
        # print("Zoom!!")
        plt.get_current_fig_manager().window.state('zoomed')



def plot_signal_interactive(x : range|list , y : list , style : str = 'b' , x_label : str = "" , y_label : str = "" , title : str = ""):
    '''
    Function plots 2D graphs with limited interactive functionalities
    '''
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)
    ax.plot(x , y , style)
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
