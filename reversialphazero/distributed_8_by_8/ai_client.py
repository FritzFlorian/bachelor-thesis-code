# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()
from reversialphazero.command_line_interface import CommandLineInterface


def main():
    weights_file = './final-long-running-test/checkpoint-00062.zip'
    nn_name = 'reversialphazero.distributed_8_by_8.neural_network.SimpleNeuralNetwork'

    command_line_interface = CommandLineInterface(ai_client=True, nn_class_name=nn_name,
                                                  ai_client_weights_file=weights_file, name='AI-Client')
    command_line_interface.parse_args()
    command_line_interface.execute()


if __name__ == "__main__":
    main()
