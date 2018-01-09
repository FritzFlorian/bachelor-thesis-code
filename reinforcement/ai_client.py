"""The actual, executable AI Client to play a match.
This can not live in core to prevent cyclic dependencies."""
import reinforcement.core as core
import reversi.client
import reinforcement.nn_client
import concurrent.futures
import logging
import time


class AIClient:
    def __init__(self, group_number, nn_class_name, weights_file, host, port):
        self.nn_class_name = nn_class_name
        self.weights_file = weights_file

        self.nn_client = reinforcement.nn_client.NeuralNetworkClient('tcp://localhost:6001')
        self.threadpool = None

        self.reversi_client = reversi.client.Client(group_number, self._find_move, host=host, port=port)

    def run(self):
        logging.info('starting NN and loading weights...')
        # FIXME: Use random port/ search for available port for nn server or use one NN server for all running clients.
        reinforcement.nn_client.start_nn_server(6001, self.nn_class_name, nn_init_args=(), batch_size=6, log_dir=None)
        self.nn_client.start()
        with open(self.weights_file, 'rb') as file:
            self.nn_client.load_weights(file.read())
        self.threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=16)

        self.reversi_client.start()
        self.reversi_client.join()

        logging.info('stopping NN...')
        self.threadpool.shutdown()
        self.nn_client.shutdown_server()
        self.nn_client.stop()

    def _find_move(self, game_state, time_limit, depth_limit):
        # ignore time limit, always use a minimum of 1 second to execute.
        # can be used to test against other, depth based ai's.
        if time_limit < 1000:
            time_limit = 1000
        time_limit = time_limit / 1000  # To seconds

        end_time = time.time() + time_limit - 0.6  # End time with some buffer time for returning the result

        # Execute the MCTS
        mcts_executor = core.MCTSExecutor(game_state, self.nn_client, thread_pool=self.threadpool)
        while end_time > time.time():
            mcts_executor.run(48)

        print('Game Prediction: {}'.format(mcts_executor.root_node.mean_action_value[self.reversi_client.client.player]))

        # Find the best move
        selected_move = None
        best_probability = -1.0
        for move, probability in mcts_executor.move_probabilities(1).items():
            if probability > best_probability:
                best_probability = probability
                selected_move = move

        return selected_move
