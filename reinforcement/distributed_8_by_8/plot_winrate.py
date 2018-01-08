import pyximport; pyximport.install()

import reinforcement.distribution as dist
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    work_dir = 'final-long-running-test'

    # Plot whole range without lines
    plot(work_dir, 0, -1, False)

    # Plot only 'good' middle part
    # Start of 'good' middle is data point 61 (checkpoint file 62, as there index starts at 1), to data point 92
    plot(work_dir, 30, 130, True)

    print_avg_winrate(work_dir, 0, -1)
    print_avg_winrate(work_dir, 61, 92)


def print_avg_winrate(work_dir, lower_bound, upper_bound):
    progress = dist.TrainingRunProgress(os.path.join(work_dir, 'stats.json'))
    progress.load_stats()

    wins = []
    for iteration in progress.stats.iterations[lower_bound:upper_bound]:
        if iteration.ai_eval.n_games < 14:
            break

        wins.append((iteration.ai_eval.nn_score + iteration.ai_eval.n_games) / (2 * iteration.ai_eval.n_games))

    winrate = np.average(wins)
    print('Avg. Winnrate: {}'.format(winrate))


def plot(work_dir, lower_bound, upper_bound, show_progress_lines, smoothing=0.9):
    x_scaling = 1 / 1000  # Show thousands on x axis

    progress = dist.TrainingRunProgress(os.path.join(work_dir, 'stats.json'))
    progress.load_stats()

    wins = []
    x_steps = []

    new_was_better = []
    for iteration in progress.stats.iterations[lower_bound:upper_bound]:
        if iteration.ai_eval.n_games < 14:
            break

        wins.append((iteration.ai_eval.nn_score + iteration.ai_eval.n_games) / (2 * iteration.ai_eval.n_games))
        x_steps.append(iteration.ai_eval.end_batch * x_scaling)

        if iteration.self_eval.new_better:
            new_was_better.append(iteration.self_eval.end_batch * x_scaling)

    smoothed_wins = [wins[0]]
    for i in range(1, len(wins)):
        smoothed_wins.append(smoothing * smoothed_wins[i - 1] + (1 - smoothing) * wins[i])

    plt.plot(x_steps, wins, linestyle='--')
    plt.plot(x_steps, smoothed_wins)
    plt.ylim(0, 1)
    plt.xlabel('Batch (in 1000)')
    plt.ylabel('% gewonnener Spiele')
    plt.axhline(y=0.5, color='r')
    if show_progress_lines:
        for better_x in new_was_better:
            plt.axvline(x=better_x, linestyle=':')
    plt.show()


if __name__ == '__main__':
    main()
