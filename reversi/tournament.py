import subprocess
from reversi.game_core import Board
from reversi.server import Server
import concurrent.futures
import time
import multiprocessing


class Client:
    def start(self, host, port):
        raise NotImplementedError('Child classes must overwrite this method')


class BinaryClient(Client):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def start(self, host, port):
        subprocess.Popen([self.path, '-i', host, '-p', str(self.port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TrivialAIClient(Client):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def start(self, host, port):
        subprocess.Popen([self.path, '-s', '{}:{}'.format(host, port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class JavaClient(Client):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def start(self, host, port):
        subprocess.Popen(['java', '-jar', self.path, '-i', host, '-p', str(port)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class Match:
    def __init__(self, map_path, port, time_limit, depth_limit, clients, group_to_player):
        self.map_path = map_path
        with open(map_path) as map_file:
            self.board = Board(map_file.read())
        self.server = Server(self.board, time_limit, depth_limit, port, group_to_player)

        self.clients = clients
        self.port = port

    def run(self, host):
        self.server.start()
        for client in self.clients:
            client.start(host, self.port)
        self.server.join()

        return self


if __name__ == '__main__':
    # Test tournament, used to show how data gathering could be done using the Match class and thread pools
    matches = []
    client_one = TrivialAIClient('./test_clients/ai_trivial')
    client_two = JavaClient('./test_clients/client.jar')

    for i in range(0, 6):
        match_one = Match('./test_maps/norm10x10.map', 2500 + i * 3, 0, i + 1, [
                            client_one,
                            client_two
                          ], None)
        match_two = Match('./test_maps/norm12x12.map', 2501 + i * 3, 0, i + 1, [
                            client_one,
                            client_one
                          ], None)

        match_three = Match('./test_maps/norm12x12.map', 2502 + i * 3, 0, i + 1, [
                             client_two,
                             client_two
                          ], None)

        matches.append(match_one)
        matches.append(match_two)
        matches.append(match_three)

    def done_callback(future):
        match = future.result()
        print('Finished Match {}'.format(match))
        match.server.history.write("{}.json".format(time.time()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(match.run, 'localhost') for match in matches]
        for future in futures:
            future.add_done_callback(done_callback)
