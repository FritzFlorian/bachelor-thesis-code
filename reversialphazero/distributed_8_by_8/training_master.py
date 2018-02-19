import pyximport; pyximport.install()
from reversialphazero.command_line_interface import CommandLineInterface
import os

work_dir = os.path.join(os.path.curdir, 'test')
nn_name = 'reversialphazero.distributed_8_by_8.neural_network.SimpleNeuralNetwork'
command_line_interface = CommandLineInterface('MASTER', training_master=True, training_work_directory=work_dir,
                                              training_maps_directory='./maps', training_master_hostname='127.0.0.1',
                                              nn_class_name=nn_name)
command_line_interface.prepare_logger()


def main():
    command_line_interface.parse_args()
    command_line_interface.adjust_settings = adjust_settings
    command_line_interface.execute()


# Manually adjust some settings for this run
def adjust_settings(stats):
    stats.settings.batch_size = 64


if __name__ == '__main__':
    main()
