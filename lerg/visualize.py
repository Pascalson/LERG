import numpy as np
import matplotlib.pyplot as plt

def plot_interactions(phi_map,x,y):
    values = np.around([[phi_map[(i,j)].item() for i in range(len(x))] for j in range(len(y))], decimals=2)
    fig = plt.figure()
    ax = plt.axes()
    im = ax.imshow(values, cmap=plt.get_cmap('Reds'))

    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticklabels(x, fontsize=11)
    ax.set_yticklabels(y, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, values[i, j],
                           ha="center", va="center", color="w")
    return fig, ax
