import pure_mcts_reinforcement.core as core
import multiprocessing


class ParallelSelfplayPool:
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


class ParallelSelfplayEvaluationPool:
    """Parallel execution of selfplay evaluation games using multiple processes on this host machine."""
    def __init__(self, maps, neural_network_one, neural_network_two, model_file_one, model_file_two,
                 n_games, simulations_per_turn=100, batch_size=4, pool_size=multiprocessing.cpu_count(),
                 port_one=6001, port_two=6002):
        self.nn_server_one = core.NeuralNetworkExecutorServer(neural_network_one, model_file_one,
                                                              batch_size=batch_size, port=port_one)
        self.nn_server_two = core.NeuralNetworkExecutorServer(neural_network_two, model_file_two,
                                                              batch_size=batch_size, port=port_two)

        self.port_one = port_one
        self.port_two = port_two

        self.n_games = n_games
        self.n_simulations = simulations_per_turn
        self.maps = maps

        self.pool_size = pool_size

    def run(self):
        self.nn_server_one.start()
        self.nn_server_two.start()

        selfplay_pool = multiprocessing.Pool(processes=self.pool_size)
        games_per_tournament = round(self.n_games / self.pool_size) + 1

        parameters =[]
        for _ in range(self.pool_size):
            nn_client_one = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port_one))
            nn_client_two = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port_two))

            parameters.append((self.maps, games_per_tournament, self.n_simulations, nn_client_one, nn_client_two))

        results = selfplay_pool.map(ParallelSelfplayEvaluationPool.play_tournament, parameters)
        selfplay_pool.close()
        selfplay_pool.join()

        result_sum = [0, 0]
        for result in results:
            result_sum[0] = result_sum[0] + result[0]
            result_sum[1] = result_sum[1] + result[1]

        actual_games = self.pool_size * games_per_tournament
        avg_result = [result_sum[0] / actual_games, result_sum[1] / actual_games]

        self.nn_server_one.stop()
        self.nn_server_two.stop()

        return avg_result

    @staticmethod
    def play_tournament(args):
        maps, n_games, n_simulations, nn_client_one, nn_client_two = args

        nn_client_one.start()
        nn_client_two.start()

        tournament = core.ModelEvaluator(nn_client_one, nn_client_two, maps)
        result = tournament.run(n_games, n_simulations)

        nn_client_one.stop()
        nn_client_two.stop()

        return result


class ParallelAITrivialPool:
    """Parallel execution of tournament games against ai trivial using multiple processes on this host machine."""

    def __init__(self, maps, neural_network, model_file, n_games, time, batch_size=4,
                 pool_size=multiprocessing.cpu_count(), port=60012):
        self.maps = maps
        self.n_games = n_games
        self.time = time
        self.pool_size = pool_size
        self.port = port

        self.nn_server = core.NeuralNetworkExecutorServer(neural_network, model_file, batch_size=batch_size, port=port)

    def run(self):
        self.nn_server.start()

        selfplay_pool = multiprocessing.Pool(processes=self.pool_size)
        games_per_tournament = round(self.n_games / self.pool_size) + 1

        parameters = []
        for _ in range(self.pool_size):
            nn_client = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port))
            parameters.append((self.maps, games_per_tournament, self.time, nn_client))

        results = selfplay_pool.map(ParallelAITrivialPool.play_tournament, parameters)
        selfplay_pool.close()
        selfplay_pool.join()

        scores_sum = [0, 0]
        stones_sum = [0, 0]
        for result in results:
            for i in range(2):
                scores_sum[i] = scores_sum[i] + result[0][i]
                stones_sum[i] = stones_sum[i] + result[1][i]

        self.nn_server.stop()

        return scores_sum, stones_sum

    @staticmethod
    def play_tournament(args):
        maps, n_games, time, nn_client = args

        nn_client.start()
        tournament = core.AITrivialEvaluator(nn_client, maps)
        result = tournament.run(n_games, time)
        nn_client.stop()

        return result
