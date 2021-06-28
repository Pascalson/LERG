import numpy as np
import matplotlib.pyplot as plt

def plot_interactions(phi_map,x,y,save_path=None):
    values = np.around([[phi_map[(i,j)].item() for i in range(len(x))] for j in range(len(y))], decimals=2)
    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=plt.get_cmap('Reds'))

    # show all ticks
    ax.set_xticks(np.arange(len(x)))
    ax.set_yticks(np.arange(len(y)))
    # label them with the respective list entries
    x = [w[:-4] for w in x]
    y = [w[:-4] for w in y]
    ax.set_xticklabels(x, fontsize=16)
    ax.set_yticklabels(y, fontsize=16)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # loop over data dimensions and create text annotations.
    for i in range(len(y)):
        for j in range(len(x)):
            text = ax.text(j, i, values[i, j],
                           ha="center", va="center", color="w")
    
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
