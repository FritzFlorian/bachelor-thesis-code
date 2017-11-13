import pure_mcts_reinforcement.core as core
import multiprocessing


class ParallelSelfplayPool():
    """Parallel execution of selfplay games using multiple processes on this host machine."""
    def __init__(self, game_state, neural_network, model_file, n_games, callback, simulations_per_turn=100,
                 batch_size=4, pool_size=multiprocessing.cpu_count(), port=6001):
        self.port = port
        self.pool_size = pool_size
        self.n_games = n_games
        self.game_state = game_state
        self.simulations_per_turn = simulations_per_turn
        self.callback = callback
        self.nn_executor_server = \
            core.NeuralNetworkExecutorServer(neural_network, model_file, batch_size=batch_size, port=port)

    def run(self):
        self.nn_executor_server.start()
        selfplay_pool = multiprocessing.Pool(processes=self.pool_size)

        for _ in range(self.n_games):
            nn_executor_client = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port))
            params = (self.game_state, nn_executor_client, self.simulations_per_turn)
            selfplay_pool.apply_async(ParallelSelfplayPool.play_game, params, callback=self.callback)

        selfplay_pool.close()
        selfplay_pool.join()
        self.nn_executor_server.stop()

    @staticmethod
    def play_game(game_state, nn_executor, n_simulations):
        nn_executor.start()
        selfplay_executor = core.SelfplayExecutor(game_state, nn_executor, n_simulations)
        result = selfplay_executor.run()
        nn_executor.stop()
        return result
