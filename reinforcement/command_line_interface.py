"""
Helper classes that allow to invoke training and execution of clients using command line arguments.
"""
# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()
import definitions
import argparse
import importlib
import logging
import reinforcement.distribution as distribution
import os
from reversi.game_core import Board


def main():
    command_line_interface = CommandLineInterface()
    command_line_interface.parse_args()
    command_line_interface.execute()


class CommandLineInterface:
    """Parses command line actions and runs the given action.
    Can be configured with default settings for specific test runs."""

    def __init__(self, nn_class_name=None, training_work_directory=None, ai_client=False, ai_client_weights_file=None,
                 match_server_hostname=definitions.REVERSI_MATCH_SERVER_DEFAULT_HOST,
                 match_server_port=definitions.REVERSI_MATCH_SERVER_DEFAULT_PORT, training_master_hostname=None,
                 training_master_port=definitions.TRAINING_MASTER_PORT, training_maps_directory=None,
                 training_master=False, selfplay_slave=False, training_map_directory=None):
        """Specify default arguments for specific test runs.

        @param nn_class_name: The Neural Network Subclass (full name) to use for this run
        @param training_work_directory: The directory to keep training progress
        @param ai_client: Set to true to run in AI client mode (to play a match)
        @param ai_client_weights_file: NN weights file to use for executing the AI client
        @param match_server_hostname: The hostname of the match/tournament server when executing the AI client
        @param match_server_port: The port of the match/tournament server when executing the AI client
        @param training_master_hostname: The hostname of the training master server
        @param training_master_port: The port the training master runs on
        @param training_master: Set to true to run as training master (performs training, depends on selfplay slaves)
        @param selfplay_slave: Set to true to run as selfplay slave (runs selfplay games, reporst to a training master)
        @param training_maps_directory: The directory with all maps to be used for the training run
        """
        self.parser = argparse.ArgumentParser(description='AI Client for Reversi. Allows playing games and training Clients.')
        self.parser.add_argument('-nn', '--nn-class-name', type=str, default=nn_class_name, dest='nn_class_name',
                                 help='The Neural Network Subclass (full name) to use for this run')
        self.parser.add_argument('-d', '--training-work-directory', type=str, default=training_work_directory, dest='training_work_directory',
                                 help='The directory to keep training progress')
        self.parser.add_argument('-ai', '--ai-client', default=ai_client, dest='ai_client', nargs='?', type=self.str2bool, const=True,
                                 help='Set to true to run in AI client mode (to play a match)')
        self.parser.add_argument('-w', '--ai-client-weights', default=ai_client_weights_file, dest='ai_client_weights',
                                 help='NN weights file to use for executing the AI client')
        self.parser.add_argument('-i', '--match-host', default=match_server_hostname, dest='match_host',
                                 help='The hostname of the match/tournament server when executing the AI client')
        self.parser.add_argument('-p', '--match-port', default=match_server_port, dest='match_port',
                                 help='The port of the match/tournament server when executing the AI client')
        self.parser.add_argument('-m', '--training-master', default=training_master, dest='training_master', nargs='?', type=self.str2bool, const=True,
                                 help='Set to true to run as training master (performs training, depends on selfplay slaves)')
        self.parser.add_argument('-s', '--selfplay-slave', default=selfplay_slave, dest='selfplay_slave', nargs='?', type=self.str2bool, const=True,
                                 help='Set to true to run as selfplay slave (runs selfplay games, reporst to a training master)')
        self.parser.add_argument('-mi', '--master-host', default=training_master_hostname, dest='master_host',
                                 help='The hostname of the training master server')
        self.parser.add_argument('-mp', '--master-port', default=training_master_port, dest='master_port',
                                 help='The port the training master runs on')
        self.parser.add_argument('-tm', '--training-maps-directory', default=training_maps_directory, dest='training_maps_directory',
                                 help='The directory with all maps to be used for the training run')

    @staticmethod
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    def parse_args(self):
        """Reads the arguments provided on the command line and parses them."""
        args = self.parser.parse_args()
        if args.ai_client:
            self.mode = 'ai-client'
            self.port = args.match_port
            self.host = args.match_host
            self.weights_file = args.ai_client_weights
            self._parse_nn_class(args)

            if not self.port:
                print('You must specify the "--match-port"!')
                exit(1)
            if not self.host:
                print('You must specify the "--match-host"!')
                exit(1)
            if not self.weights_file:
                print('You must specify the "--ai-client-weights" file!')
                exit(1)
        elif args.training_master:
            self.mode = 'training-master'
            self.port = args.master_port
            self.work_dir = args.training_work_directory
            self.maps_dir = args.training_maps_directory
            self._parse_nn_class(args)

            if not self.port:
                print('You must specify the "--master-port"!')
                exit(1)
            if not self.work_dir:
                print('You must specify the "--training-work-directory" path!')
                exit(1)
            if not self.maps_dir:
                print('You must specify the "--training-maps-directory" path!')
                exit(1)
        elif args.selfplay_slave:
            self.mode = 'selfplay-slave'
            self.port = args.master_port
            self.host = args.master_host

            if not self.port:
                print('You must specify the "--master-port"!')
                exit(1)
            if not self.host:
                print('You must specify the "--master-host"!')
                exit(1)
        else:
            print('You must choose one of "--ai-client", "--training-master" or "--selfplay-slave"')
            exit(1)

    def _parse_nn_class(self, args):
        nn_class_name = args.nn_class_name

        if nn_class_name is None:
            print('"--nn-class-name" has to be set to be set!')
        self.nn_class_name = nn_class_name

        module_name, class_name = nn_class_name.rsplit('.', 1)
        try:
            nn_module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            print('NN Class Module not found, be sure to specify the FULLY QUALIFIED name of it.')
            exit(1)

        try:
            self.nn_class = getattr(nn_module, class_name)
        except AttributeError:
            print('NN Class not found in given module, be sure to specify the FULLY QUALIFIED name of it.')
            exit(1)

    def execute(self):
        """Executes the programm configured to the given arguments. Call parse_args first!"""
        if self.mode == 'ai-client':
            self._execute_ai_client()
        elif self.mode == 'training-master':
            self._execute_training_master()
        elif self.mode == 'selfplay-slave':
            self._execute_selfplay_slave()

    def _execute_ai_client(self):
        print('Executing AI Client to play on match-server "{}:{}".'.format(self.host, self.port))

    def _execute_training_master(self):
        print('Executing Training Master with maps directory "{}", work directory "{}", on port "{}".'.format(self.maps_dir, self.work_dir, self.port))
        print('Please Connect Selfplay Slave Nodes to this training master.')
        print('Shut down with Control-C ONCE!')

        boards = []
        for map_name in os.listdir(self.maps_dir):
            map_path = os.path.join(self.maps_dir, map_name)
            if map_path.endswith('.map'):
                print('Loading Map "{}"'.format(map_path))
                with open(map_path) as file:
                    boards.append(Board(file.read()))

        if len(boards) <= 0:
            print('No maps (ending with .map) fond in maps directory ({}).'.format(self.maps_dir))

        logging.basicConfig(level=logging.DEBUG)
        training_master = distribution.TrainingMaster(self.work_dir, self.nn_class_name, boards)
        training_master.run()

    def _execute_selfplay_slave(self):
        print('Executing Selfplay Slave with Training Master at "{}:{}".'.format(self.host, self.port))
        print('Shut down with Control-C ONCE!')

        logging.basicConfig(level=logging.DEBUG)
        selfplay_server = distribution.PlayingSlave('tcp://{}:{}'.format(self.host, self.port))
        selfplay_server.run()


if __name__ == '__main__':
    main()
