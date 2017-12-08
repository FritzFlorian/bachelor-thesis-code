from reversi.client import Client
import logging
import concurrent.futures
import time
import reinforcement.core as core
import reinforcement.simple_8_by_8.simple_8_by_8 as simple_8_by_8


def main():
    logging.basicConfig(level=logging.INFO)

    threadpool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    nn_server = core.NeuralNetworkExecutorServer(simple_8_by_8.SimpleNeuralNetwork(),
                                                 './run_final_8_by_8/best-checkpoint/checkpoint.ckpt',
                                                 batch_size=4, port=6001)
    nn_server.start()
    nn_client = core.NeuralNetworkExecutorClient('tcp://localhost:6001')
    nn_client.start()

    time.sleep(10)

    def find_move(game_state, time_limit, depth_limit):
        end_time = time.time() + (time_limit / 1000) - 0.5

        # Execute the MCTS
        mcts_executor = core.MCTSExecutor(game_state, nn_client, thread_executor=threadpool)
        while end_time > time.time():
            mcts_executor.run(32)

        # Find the best move
        selected_move = None
        best_probability = -1.0
        for move, probability in mcts_executor.move_probabilities(1).items():
            if probability > best_probability:
                best_probability = probability
                selected_move = move

        return selected_move

    client = Client(14, find_move)
    client.start()
    client.join()

    threadpool.shutdown(False)
    nn_client.stop()
    nn_server.stop()


if __name__ == '__main__':
    main()
