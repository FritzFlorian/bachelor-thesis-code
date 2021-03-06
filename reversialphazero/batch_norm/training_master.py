import pyximport; pyximport.install()
from reversialphazero.command_line_interface import CommandLineInterface
import os


work_dir = os.path.join(os.path.curdir, 'test')
nn_name = 'reversialphazero.batch_norm.neural_network.SimpleNeuralNetwork'

command_line_interface = CommandLineInterface('MASTER', training_master=True, training_work_directory=work_dir,
                                              training_maps_directory='./maps', training_master_hostname='127.0.0.1',
                                              nn_class_name=nn_name)
command_line_interface.prepare_logger()


def main():
    command_line_interface.parse_args()
    command_line_interface.config._n_self_play = 77
    command_line_interface.config._n_ai_eval = 21
    command_line_interface.config._n_self_eval = 21
    command_line_interface.config._training_history_size = 200
    command_line_interface.config._batch_size = 200
    command_line_interface.config._c_puct = 1
    command_line_interface.config._simulations_per_turn = 512  # 4 times the simulations per training run
    command_line_interface.execute()


if __name__ == '__main__':
    main()
