import matplotlib.pyplot as plt
import numpy as np
import os


def plot_qp_results(path, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='gold', marker='o', label='Final candidate')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    ax.view_init(45, 45)
    plt.savefig(os.path.dirname(__file__) + f"/../plots/{title}")
    plt.show()


def plot_lp_results(path, title):
    fig, ax = plt.subplots(1, 1)
    path = np.array(path)

    x = np.linspace(-1, 3, 1000)
    y = np.linspace(-2, 2, 1000)
    inequalities = {
        'y=0': (x, x * 0),
        'y=1': (x, x * 0 + 1),
        'x=2': (y * 0 + 2, y),
        'y=-x+1': (x, -x + 1)
    }

    for label, (x_vals, y_vals) in inequalities.items():
        ax.plot(x_vals, y_vals, label=label)

    ax.fill([0, 2, 2, 1], [1, 1, 0, 0], 'lightgray', label='Feasible region')
    ax.plot(path[:, 0], path[:, 1], c='k', label='Path')
    ax.scatter(path[-1][0], path[-1][1], s=50, c='gold', marker='o', label='Final candidate')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    plt.savefig(os.path.dirname(__file__) + f"/../plots/{title}")
    plt.show()
