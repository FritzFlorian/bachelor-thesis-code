from reversi.network_core import BasicClient, DEFAULT_HOST, DEFAULT_PORT
from reversi.game_core import GameState
import reversi.network_core as network_core
import threading
import random
import logging


class Client(threading.Thread):
    def __init__(self, group, find_move, host=DEFAULT_HOST, port=DEFAULT_PORT):
        super().__init__()
        self.logger = logging.getLogger("Client ({})".format(group))
        self.client = BasicClient(group, host, port)
        self.find_move = find_move
        self.game = None

    def run(self):
        self.client.start()
        self.game = GameState(self.client.board)

        self._game_loop()
        self.logger.info("Game Ended")

        self.client.stop()

    def _game_loop(self):
        while True:
            message = self.client.read_message()
            if isinstance(message, network_core.EndPhaseTwoMessage):
                return
            elif isinstance(message, network_core.EndPhaseOneMessage):
                self.game.bomb_phase = True
                self.logger.info("Phase One Ended")
            elif isinstance(message, network_core.MoveRequestMessage):
                self.logger.info("Move Request from server ({}, {})".format(message.time_limit, message.depth_limit))
                (player, pos, choice) = self.find_move(self.game, message.time_limit, message.depth_limit)
                self.logger.info("Answer: {}, {}".format(pos, choice))
                move_message = network_core.MoveResponseMessage(pos, choice)
                self.client.send_message(move_message)
            elif isinstance(message, network_core.DisqualificationMessage):
                self.logger.info("Player {} Disqualified!".format(message.player))
                self.game.disqualify_player(message.player)
                if message.player == self.client.player:
                    self.logger.info("Client was disqualified, shutting down...")
                    return
            elif isinstance(message, network_core.MoveNotificationMessage):
                self.game = self.game.execute_move(message.player, message.pos, message.choice)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def find_move(game, time_limit, depth_limit):
        possible = game.get_next_possible_moves()
        return random.choice(possible).last_move

    client = Client(14, find_move)
    client.start()
    client.join()
