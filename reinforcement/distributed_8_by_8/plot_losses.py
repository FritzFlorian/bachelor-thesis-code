import pyximport; pyximport.install()

import reinforcement.distribution as dist
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    work_dir = 'final-long-running-test'

    # Plot general loss
    plot(os.path.join(work_dir, 'loss.csv'), 0, -1, loss_label='Gesamter Fehler')

    # Plot reg loss
    plot(os.path.join(work_dir, 'loss_reg.csv'), 0, -1, loss_label='L2 Regulierungs Fehler', y_lim=(0.49, 0.55))
    plot(os.path.join(work_dir, 'loss_reg.csv'), 100, 500, loss_label='L2 Regulierungs Fehler', y_lim=(0.49, 0.55))


def plot(stats_file, lower_bound, upper_bound, smoothing=0.9, loss_label='Fehler', y_lim=None):
    x_scaling = 1 / 1000  # Show thousands on x axis

    stats = np.loadtxt(stats_file, skiprows=1, delimiter=',')
    stats = np.transpose(stats)  # Change axis for easy selection
    x_steps = stats[1][lower_bound:upper_bound] * x_scaling
    y_loss = stats[2][lower_bound:upper_bound]

    smoothed_loss = [y_loss[0]]
    for i in range(1, len(y_loss)):
        smoothed_loss.append(smoothing * smoothed_loss[i - 1] + (1 - smoothing) * y_loss[i])

    plt.plot(x_steps, y_loss, alpha=0.4)
    plt.plot(x_steps, smoothed_loss)
    if y_lim:
        plt.ylim(y_lim)
    plt.xlabel('Batch (in 1000)')
    plt.ylabel(loss_label)

    # Markers for important part
    plt.axvline(x=1500000 * x_scaling, color='r')
    plt.axvline(x=2100000 * x_scaling, color='r')

    plt.show()


if __name__ == '__main__':
    main()
