"""The actual, executable AI Client to play a match.
This can not live in core to prevent cyclic dependencies."""
import hometrainer.util
import hometrainer.neural_network
import hometrainer.agents
import reversi.network
import reversialphazero.ai_trivial_agent
import reversialphazero.core
import logging


class AIClient:
    def __init__(self, group_number, nn_class_name, weights_file, host, port, config):
        self.config = config
        self.nn_class_name = nn_class_name
        self.weights_file = weights_file

        # Keep an internal NN client to load the correct weights
        self.nn_client = hometrainer.neural_network.NeuralNetworkClient('tcp://localhost:6001')

        # Keep a ReversiXT Client to communicate with the server
        self.reversi_client = reversi.network.Client(group_number, self._find_move, host=host, port=port,
                                                game_start_callback=self._game_start, game_end_callback=self._game_end,
                                                move_callback=self._move_executed)

        self.nn_agent = hometrainer.agents.NeuralNetworkAgent(hometrainer.util.deepcopy(self.nn_client),
                                                              config=self.config)

    def run(self):
        logging.info('starting NN and loading weights...')
        # FIXME: Use random port/ search for available port for nn server or use one NN server for all running clients.
        # Start a neural network server and load weights
        hometrainer.neural_network.start_nn_server(6001, self.nn_class_name, config=self.config,
                                                   nn_init_args=(), batch_size=6, log_dir=None)
        self.nn_client.start(self.config)
        with open(self.weights_file, 'rb') as file:
            self.nn_client.load_weights(file.read())

        self.reversi_client.start()
        self.reversi_client.join()

        logging.info('stopping NN...')
        self.nn_client.shutdown_server()
        self.nn_client.stop()

    def _game_start(self, game_state, player):
        game_state = reversialphazero.core.ReversiGameState(game_state)
        self.nn_agent.game_start(game_state, player)

    def _game_end(self, game_state):
        game_state = reversialphazero.core.ReversiGameState(game_state)
        self.nn_agent.game_ended(game_state)

    def _find_move(self, game_state, time_limit, depth_limit):
        game_state = reversialphazero.core.ReversiGameState(game_state)
        time_limit = time_limit - 500  # Be sure to return in time...we remove 500ms from our time.

        if time_limit > 0:
            return self.nn_agent.find_move_with_time_limit(game_state, round(time_limit / 1000)).internal_tuple
        else:
            # Make up some multiplier, as depth limit is different to our simulation limit
            return self.nn_agent.find_move_with_iteration_limit(game_state, depth_limit * 50).internal_tuple

    def _move_executed(self, old_game_state, move, new_game_state):
        move = reversialphazero.core.ReversiMove(move)
        old_game_state = reversialphazero.core.ReversiGameState(old_game_state)
        new_game_state = reversialphazero.core.ReversiGameState(new_game_state)

        self.nn_agent.move_executed(old_game_state, move, new_game_state)
