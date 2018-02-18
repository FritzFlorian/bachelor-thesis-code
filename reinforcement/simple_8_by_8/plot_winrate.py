import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import numpy as np
import io


def main():
    with open('run_final_8_by_8/avg_score.png', 'wb') as file:
        plot = plot_external_eval_avg_score('run_final_8_by_8/stats.csv', 0, -1, False, smoothing=0.15,
                                            x_label='Iteration', y_label='Durschnittlicher Spielausgang',
                                            score_label='Spielausgang',
                                            smoothed_score_lable='geglätteter Spielausgang')
        file.write(plot.getbuffer())


def plot_external_eval_avg_score(stats_file, lower_bound, upper_bound, show_progress_lines,
                                 smoothing=0.1, min=-1, max=1, middle=0,
                                 x_label='Batch', y_label='Average Game Outcome',
                                 smoothed_score_lable='Smoothed Score', score_label='Score'):
    """Returns a binary containing a plot of the current winrate."""
    n_games = 21

    stats = np.loadtxt(stats_file, skiprows=1, delimiter=',')

    avg_score = []
    x_steps = []

    for iteration in stats[lower_bound:upper_bound]:
        avg_score.append((iteration[1] - iteration[2]) / n_games)
        x_steps.append(iteration[0] + 1)

    smoothed_wins = smooth_array(avg_score, smoothing)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    label_1, = plt.plot(x_steps, avg_score, linestyle='--', label=score_label)
    label_2, = plt.plot(x_steps, smoothed_wins, label=smoothed_score_lable)
    plt.legend(handles=[label_1, label_2])
    plt.ylim(min, max)
    plt.xlim(x_steps[0] - 1, x_steps[-1] + 1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(y=middle, color='r')

    result = io.BytesIO()
    plt.savefig(result, dpi=400)
    plt.clf()

    return result


def smooth_array(array, smoothing):
    smoothed_array = []
    max_steps = max(1, round(len(array) * smoothing))
    for i in range(len(array)):
        values = [array[i]]
        for j in range(1, max_steps + 1):
            if i - j < 0:
                break
            values.append(array[i - j])
        for j in range(1, max_steps + 1):
            if i + j >= len(array):
                break
            values.append(array[i + j])

        smoothed_array.append(np.average(values))

    return smoothed_array


if __name__ == '__main__':
    main()
