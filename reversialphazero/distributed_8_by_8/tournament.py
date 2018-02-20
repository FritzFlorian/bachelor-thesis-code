import pyximport; pyximport.install()
import reversialphazero.config
import reversialphazero.ai_trivial_agent
import reversialphazero.ai_java_agent
import hometrainer.agents
import hometrainer.neural_network
import hometrainer.util
import hometrainer.executors
import reversi.game_core
import reversialphazero.core


simple_8_by_8_board = reversi.game_core.Board("""\
2
0
0 0
8 8
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 1 2 0 0 0
0 0 0 2 1 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
""")


def main():
    time_limits = [1, 2, 5, 10]
    n_games = 20
    start_game_state = reversialphazero.core.ReversiGameState(reversi.game_core.GameState(simple_8_by_8_board))

    config = reversialphazero.config.CustomConfiguration()
    nn_client = hometrainer.neural_network.NeuralNetworkClient('tcp://localhost:7500')
    init_nn(nn_client, config)

    ai_nn_client = hometrainer.agents.NeuralNetworkAgent(nn_client, config)
    ai_trivial_client = reversialphazero.ai_trivial_agent.AITrivialAgent(time_bonus=0)
    ai_java_client = reversialphazero.ai_java_agent.AIJavaAgent(time_bonus=0)

    with open('tournament.log', 'w') as log:
        def write_log(string):
            log.write(string + '\n')
            log.flush()
            print(string)

        # NN vs Java
        for time_limit in time_limits:
            write_log('Start run with time limit {}'.format(time_limit))
            for game in range(n_games):
                match_executor = hometrainer.executors.GameExecutor(start_game_state, [ai_nn_client, ai_java_client],
                                                                    add_up_time=True)
                results = match_executor.play_game(time_limit=time_limit)
                write_log('NN: {} vs Java: {}'.format(results[ai_nn_client], results[ai_java_client]))

        # NN vs Trivial
        for time_limit in time_limits:
            write_log('Start run with time limit {}'.format(time_limit))
            for game in range(n_games):
                match_executor = hometrainer.executors.GameExecutor(start_game_state, [ai_nn_client, ai_trivial_client],
                                                                    add_up_time=True)
                results = match_executor.play_game(time_limit=time_limit)
                write_log('NN: {} vs Trivial: {}'.format(results[ai_nn_client], results[ai_trivial_client]))

    stop_nn(nn_client, config)


def init_nn(nn_client, config):
    nn_class_name = 'reversialphazero.distributed_8_by_8.neural_network.SimpleNeuralNetwork'
    weighs_file = 'final-long-running-test/checkpoint-00062.zip'

    local_client = hometrainer.util.deepcopy(nn_client)
    hometrainer.neural_network.start_nn_server(7500, nn_class_name, config)
    local_client.start(config)
    with open(weighs_file, 'rb') as weights_binary:
        local_client.load_weights(weights_binary.read())
    local_client.stop()


def stop_nn(nn_client, config):
    local_client = hometrainer.util.deepcopy(nn_client)
    local_client.start(config)
    local_client.shutdown_server()
    local_client.stop()


if __name__ == '__main__':
    main()
