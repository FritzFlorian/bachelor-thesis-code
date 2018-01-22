# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()
from reinforcement.command_line_interface import CommandLineInterface


CommandLineInterface.prepare_logger()


def main():
    command_line_interface = CommandLineInterface(selfplay_slave=True,
                                                  training_master_hostname='127.0.0.1')
    command_line_interface.parse_args()
    command_line_interface.execute()


if __name__ == '__main__':
    main()
