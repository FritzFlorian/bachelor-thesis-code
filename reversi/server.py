from reversi.network_core import BasicServer, DEFAULT_PORT
from reversi.game_core import GameState, Board, Field
import reversi.network_core as network_core
import threading
import logging


class Server(threading.Thread):
    def __init__(self, board, time, depth, port=DEFAULT_PORT):
        super().__init__()
        self.logger = logging.getLogger("Server ({})".format(port))
        self.game = GameState(board)
        self.server = BasicServer(board, port)
        self.time = time
        self.depth = depth

    def run(self):
        self.server.start()

        groups = []
        for i in range(self.game.board.n_players):
            self.logger.info("Waiting for {} more players to connect...".format(self.game.board.n_players - i))
            groups.append(self.server.accept_client())

        self.logger.info("All players connected, distributing maps and player numbers.")
        for i in range(self.game.board.n_players):
            self.server.set_player_for_group(groups[i], Field(chr(ord(Field.PLAYER_ONE.value) + i)))

        self.logger.info("Starting Game")
        self._game_loop()

    def _game_loop(self):
        while True:
            was_in_bomb_phase = self.game.bomb_phase
            next_moves = self.game.get_next_possible_moves()
            if len(next_moves) == 0:
                self.server.broadcast_message(network_core.EndPhaseTwoMessage())
                self.logger.info("No more moves, ending game...")
                return

            if next_moves[0].bomb_phase != was_in_bomb_phase:
                self.server.broadcast_message(network_core.EndPhaseOneMessage())

            (player, _, _) = next_moves[0].last_move
            self.logger.info("Send move request to player {} ({} possible moves).".format(player.value, len(next_moves)))

            try:
                move_request = network_core.MoveRequestMessage(1000 * self.time, self.depth)
                self.server.send_player_message(player, move_request)

                move_response = self.server.read_player_message(player, network_core.MoveResponseMessage, self.time)
                self.game = self.game.execute_move(player, move_response.pos, move_response.choice)
                self.logger.info("Player Move: ({}, {})".format(move_response.pos, move_response.choice))
                self.logger.info(self.game.board.map_string())

                move_notification = network_core.MoveNotificationMessage(move_response.pos, move_response.choice,
                                                                         player)
                self.server.broadcast_message(move_notification)
            except network_core.DisqualifiedError as err:
                self.logger.info("Player {} Disqualified! {}".format(player.value, err))
                self.game.disqualify_player(player)
                self.server.broadcast_message(network_core.DisqualificationMessage(player))

            if len(self.game.players) <= 1:
                # Everyone is disqualified, end the game
                if not self.game.bomb_phase:
                    self.server.broadcast_message(network_core.EndPhaseOneMessage())
                    self.server.broadcast_message(network_core.EndPhaseTwoMessage())
                    self.logger.info("Everyone is disqualified, ending game...")
                    return


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    board = Board("""\
    2
    0
    2 0
    6 6
    0 0 0 0 0 0
    0 0 0 0 0 0
    0 0 1 2 0 0
    0 0 2 1 0 0
    0 0 0 0 0 0
    0 0 0 0 0 0
    """)

    server = Server(board, 10, 0)
    server.start()
    server.join()
