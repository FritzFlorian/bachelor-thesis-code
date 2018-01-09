# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()
from reinforcement.command_line_interface import CommandLineInterface
import os


def main():
    work_dir = os.path.join(os.path.curdir, 'test')
    nn_name = 'reinforcement.multiple_maps.neural_network.SimpleNeuralNetwork'

    command_line_interface = CommandLineInterface(training_master=True, training_work_directory=work_dir,
                                                  training_maps_directory='./maps', training_master_hostname='127.0.0.1',
                                                  nn_class_name=nn_name)
    command_line_interface.parse_args()
    command_line_interface.execute()


if __name__ == '__main__':
    main()
