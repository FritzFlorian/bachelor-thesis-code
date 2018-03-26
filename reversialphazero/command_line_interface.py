"""
Helper classes that allow to invoke training and execution of clients using command line arguments.
"""
# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()
import definitions
import argparse
import importlib
import logging
import hometrainer.distribution as distribution
import os
from reversi.game_core import Board
import hometrainer.core
from reversialphazero.ai_client import AIClient
import hometrainer.util
import zmq
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import multiprocessing
import reversialphazero.config
import reversialphazero.core
import reversi.game_core


def main():
    command_line_interface = CommandLineInterface('MAIN-CMD')
    command_line_interface.parse_args()
    command_line_interface.prepare_logger()
    command_line_interface.execute()


class CommandLineInterface:
    """Parses command line actions and runs the given action.
    Can be configured with default settings for specific test runs."""

    def __init__(self, name, log_file='console_out.log',
                 nn_class_name=None, training_work_directory=None, ai_client=False, ai_client_weights_file=None,
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
        self.parser.add_argument('-ws', '--web-server', default=False, dest='web_server', nargs='?',
                                 type=self.str2bool, const=True,
                                 help='Set to true to run a monitoring webinterface on port {}'.format(definitions.WEB_INTERFACE_PORT))

        self.config = reversialphazero.config.CustomConfiguration()
        self.logging_client = None
        self.name = name
        self.log_file = log_file

    def prepare_logger(self, level=logging.INFO):
        """Call this at the start of your main python file (outside of the main method, has to always run).
        This will setup the logger for all started processes."""
        # This is needed for tensorflow and the logging to work properly
        multiprocessing.set_start_method('spawn', True)

        def logging_server_handler(record):
            pass

        logging_server = LoggingSever(logging_server_handler, self.config, log_file=self.log_file)
        logging_server.start()

        self.logging_client = LoggingClient(self.config)
        self.logging_client.start()

        handler = InterceptingHandler(self._local_logging_handler)
        format = '[%(asctime)-15s] %(levelname)-10s-> %(message)s'
        logging.basicConfig(format=format, level=level, handlers=[handler])

    def _local_logging_handler(self, record):
        print(record)
        self.logging_client.send_log_message('[{:15s}]{}'.format(self.name, record))

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
        self.web_server = args.web_server
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
        if self.web_server:
            self._start_webserver()

        if self.mode == 'ai-client':
            self._execute_ai_client()
        elif self.mode == 'training-master':
            self._execute_training_master()
        elif self.mode == 'selfplay-slave':
            self._execute_selfplay_slave()

    def _start_webserver(self):
        thread = threading.Thread(target=self._run_webserver)
        thread.start()

    def _run_webserver(self):
        server_address = ('0.0.0.0', definitions.WEB_INTERFACE_PORT)
        httpd = HTTPServer(server_address, MonitoringInterfaceHandler)
        httpd.output_log_file = self.log_file
        httpd.work_dir = self.work_dir
        httpd.serve_forever()

    def _execute_ai_client(self):
        print('Executing AI Client to play on match-server "{}:{}".'.format(self.host, self.port))
        ai_client = AIClient(14, self.nn_class_name, self.weights_file, self.host, self.port, self.config)
        ai_client.run()

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

        # Convert the boards to 'generic' game states accepted by hometrainer
        start_game_states = [reversi.game_core.GameState(board) for board in boards]
        start_game_states = [reversialphazero.core.ReversiGameState(game_state) for game_state in start_game_states]

        logging.basicConfig(level=logging.DEBUG)
        training_master = distribution.TrainingMaster(self.work_dir, self.nn_class_name, start_game_states, config=self.config)
        training_master.run()

    def _execute_selfplay_slave(self):
        print('Executing Selfplay Slave with Training Master at "{}:{}".'.format(self.host, self.port))
        print('Shut down with Control-C ONCE!')

        logging.basicConfig(level=logging.DEBUG)
        selfplay_server = distribution.PlayingSlave('tcp://{}:{}'.format(self.host, self.port), config=self.config)
        selfplay_server.run()


# Some Helpers for Logging/Monitoring
class InterceptingHandler(logging.Handler):
    def __init__(self, handler_method):
        super().__init__()
        self.handler_method = handler_method

    def emit(self, record):
        self.handler_method(self.format(record))


class LoggingSever:
    def __init__(self, on_message, config, port=definitions.LOGGING_SERVER_PORT, log_file=None):
        self.config = config
        self.port = port
        self.on_message = on_message

        self.log_file_name = log_file

    def start(self):
        """Try to start a logging server, returns false if port was already taken"""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.REP)
            hometrainer.util.secure_server_connection(self.socket, self.context, only_localhost=True, config=self.config)
            self.socket.bind('tcp://*:{}'.format(self.port))

            self.thread = threading.Thread(target=self._run)
            self.thread.start()
        except zmq.ZMQBaseError:
            return False

        return True

    def _run(self):
        if self.log_file_name:
            with open(self.log_file_name, 'a') as file:
                while True:
                    message = self.socket.recv_string()
                    self.socket.send_string('OK')
                    file.write(message + '\n')
                    file.flush()
                    self.on_message(message)
        else:
            while True:
                message = self.socket.recv_string()
                self.socket.send_string('OK')
                self.on_message(message)


class LoggingClient:
    def __init__(self, config, port=definitions.LOGGING_SERVER_PORT):
        self.config = config
        self.port = port

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        hometrainer.util.secure_client_connection(self.socket, self.context, self.config, only_localhost=True)
        self.socket.connect('tcp://127.0.0.1:{}'.format(self.port))

    def send_log_message(self, message):
        self.socket.send_string(message)
        self.socket.recv_string()


class MonitoringInterfaceHandler(BaseHTTPRequestHandler):
    # GET
    def do_GET(self):
        try:
            if self.path == '/graph.png':
                self._do_get_graph()
            elif self.path in {'', '/', '/index.html', '/index'}:
                self._do_get_default()
            else:
                self.send_text_response(404, 'Not Found')
        except Exception as e:
            self.send_text_response(500, 'Exception: {}'.format(e))

    def _do_get_default(self):
        log_messages = 'No Log Messages to Display'
        if self.server.output_log_file:
            log_messages = self.tail_file(self.server.output_log_file, 1000)

        progress = distribution.TrainingRunProgress(os.path.join(self.server.work_dir, 'stats.json'))
        progress.load_stats()

        current_iteration = progress.stats.progress.iteration
        n_games = hometrainer.util.count_files(os.path.join(self.server.work_dir, 'selfplay-data'))

        message = """
<!DOCTYPE html>
<html lang="en">
    <body>
        <textarea style='width: 100%; height: 500px;' id='log_textarea'>
{}
        </textarea>
        <p>Current iteration: {}</p>
        <p>Number of selfplay games: {}</p>
        <img src='graph.png' style='width: 100%; height: auto; max-width: 1000px; margin: auto; display:block;'/>
        <script>
        (function() {{
            var textarea = document.getElementById('log_textarea');
            textarea.scrollTop = textarea.scrollHeight;
        }})();
        </script>
    </body>
</html>
""".format(log_messages, current_iteration, n_games)

        self.send_text_response(200, message)

    def _do_get_graph(self):
        if not self.server.work_dir:
            self.send_response(404)
            return

        try:
            self.send_response(200)
            self.send_header('Content-type', 'image/png')
            self.end_headers()

            img = hometrainer.util.plot_external_eval_avg_score(self.server.work_dir, 0, -1, False)
            self.wfile.write(img.getbuffer())
        except Exception as e:
            # We want to be sure that this does not cause any problems
            # when having a long running training process.
            logging.error('Could not print graph for web interface! {}'.format(e))
            self.send_response(500)
            return

    def send_text_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

        self.wfile.write(bytes(message, "utf8"))

    @staticmethod
    def tail_file(filename, nlines):
        with open(filename) as qfile:
            qfile.seek(0, os.SEEK_END)
            endf = position = qfile.tell()
            linecnt = 0
            while position >= 0:
                qfile.seek(position)
                next_char = qfile.read(1)
                if next_char == "\n" and position != endf - 1:
                    linecnt += 1

                if linecnt == nlines:
                    break
                position -= 1

            if position < 0:
                qfile.seek(0)

            return qfile.read()


if __name__ == '__main__':
    main()
