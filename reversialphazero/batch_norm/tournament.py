import pyximport; pyximport.install()
import reversialphazero.config
import reversialphazero.ai_trivial_agent
import reversialphazero.ai_java_agent
import hometrainer.agents
import hometrainer.neural_network
import hometrainer.util
import hometrainer.executors
import reversi.game_core
import os
import reversialphazero.core


def main():
    time_limits = [1]
    n_games = 10
    known_start_game_states = load_start_game_states('maps')
    unknown_start_game_states = load_start_game_states('unknown_maps')

    config = reversialphazero.config.CustomConfiguration()
    nn_client = hometrainer.neural_network.NeuralNetworkClient('tcp://localhost:7500')
    init_nn(nn_client, config)

    ai_nn_client = hometrainer.agents.NeuralNetworkAgent(nn_client, config)
    ai_trivial_client = reversialphazero.ai_trivial_agent.AITrivialAgent(time_bonus=0)

    with open('tournament.log', 'w') as log:
        def write_log(string):
            log.write(string + '\n')
            log.flush()
            print(string)

        write_log('Start tests on known maps')
        for time_limit in time_limits:
            write_log('Start run with time limit {}'.format(time_limit))
            for name, game_state in known_start_game_states.items():
                write_log('Start matches on map {}'.format(name))
                for game in range(n_games):
                    match_executor = hometrainer.executors.GameExecutor(game_state, [ai_nn_client, ai_trivial_client],
                                                                        add_up_time=True)
                    results = match_executor.play_game(time_limit=time_limit)
                    write_log('NN: {} vs Trivial: {}'.format(results[ai_nn_client], results[ai_trivial_client]))

        write_log('Start tests on unknown maps')
        for time_limit in time_limits:
            write_log('Start run with time limit {}'.format(time_limit))
            for name, game_state in unknown_start_game_states.items():
                write_log('Start matches on map {}'.format(name))
                for game in range(n_games):
                    match_executor = hometrainer.executors.GameExecutor(game_state, [ai_nn_client, ai_trivial_client],
                                                                        add_up_time=True)
                    results = match_executor.play_game(time_limit=time_limit)
                    write_log('NN: {} vs Trivial: {}'.format(results[ai_nn_client], results[ai_trivial_client]))

    stop_nn(nn_client, config)


def init_nn(nn_client, config):
    nn_class_name = 'reversialphazero.normalize_probs.neural_network.SimpleNeuralNetwork'
    weighs_file = 'final-long-running-test/best-checkpoint.zip'

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


def load_start_game_states(map_directory):
    result = dict()

    for filename in os.listdir(map_directory):
        with open(os.path.join(map_directory, filename)) as map_file:
            board = reversi.game_core.Board(map_file.read())
            game_state = reversialphazero.core.ReversiGameState(reversi.game_core.GameState(board))
            result[filename] = game_state

    return result


if __name__ == '__main__':
    main()
