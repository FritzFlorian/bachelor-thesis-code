import pyximport; pyximport.install()

import hometrainer.util


def main():
    work_dir = 'final-long-running-test'

    # Plot whole range without lines
    with open('winrate.png', 'wb') as file:
        plot = hometrainer.util.plot_external_eval_avg_score(work_dir, 0, -1, False, smoothing=0.05, x_scaling=10**-6,
                                                             y_label='Durchschnittlicher Spielausgang',
                                                             x_label='Batch (in Millionen)',
                                                             score_label='Spielausgang',
                                                             smoothed_score_lable='geglätteter Spielausgang')
        file.write(plot.getbuffer())


if __name__ == '__main__':
    main()
