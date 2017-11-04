from reversi.network_core import BasicServer, DEFAULT_PORT
from reversi.game_core import GameState, Board, Field, DisqualifiedError
import reversi.network_core as network_core
import threading
import logging
import time
import reversi.serialization as serial


class Server(threading.Thread):
    def __init__(self, board, time, depth, port=DEFAULT_PORT):
        super().__init__()
        self.logger = logging.getLogger("Server ({})".format(port))
        self.game = GameState(board)
        self.server = BasicServer(board, port)
        self.time = time * 1000
        self.depth = depth
        self.history = serial.GameHistory()

        self.times = dict()
        for player in self.game.players:
            self.times[player] = 0

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

        self.server.stop()

    def _game_loop(self):
        while True:
            was_in_bomb_phase = self.game.bomb_phase
            next_moves = self.game.get_next_possible_moves()
            if len(next_moves) == 0:
                self._end_game()
                return

            if next_moves[0].bomb_phase != was_in_bomb_phase:
                self.server.broadcast_message(network_core.EndPhaseOneMessage())

            (player, _, _) = next_moves[0].last_move
            self._let_player_move(player)

            if len(self.game.players) <= 1:
                self._end_game_disqualified()
                return

    def _let_player_move(self, player):
        try:
            self._inc_player_time(player)
            self._send_move_request(player)

            start_time = time.time()
            self._process_move_answer(player)
            move_time_in_ms = int((time.time() - start_time) * 1000)
            self.times[player] = self.times[player] - move_time_in_ms
            self.logger.info("Turn took {} ms".format(move_time_in_ms))

            self._broadcast_last_move_notification()
        except DisqualifiedError as err:
            self._disqualify_player(player, err)

    def _send_move_request(self, player):
        self.logger.info("Send move request to player {} ({} ms,  depth {})."
                         .format(player.value, self.times[player], self.depth))
        move_request = network_core.MoveRequestMessage(self.times[player], self.depth)
        self.server.send_player_message(player, move_request)

    def _process_move_answer(self, player):
        move_response = \
            self.server.read_player_message(player, network_core.MoveResponseMessage, self.times[player]/1000)
        self.logger.info("Player Move: ({}, {})".format(move_response.pos, move_response.choice))

        self.game = self.game.execute_move(player, move_response.pos, move_response.choice)
        if not self.game:
            raise DisqualifiedError("Client send invalid move!", player)

        self.logger.info(self.game.board.map_string())

    def _broadcast_last_move_notification(self):
        (player, pos, choice) = self.game.last_move
        self.history.add_move(player, pos, choice)
        move_notification = network_core.MoveNotificationMessage(pos, choice, player)
        self.server.broadcast_message(move_notification)

    def _inc_player_time(self, player):
        if self.times[player] < 0:
            self.times[player] = 0
        self.times[player] = self.times[player] + self.time

    def _end_game_disqualified(self):
        if not self.game.bomb_phase:
            self.server.broadcast_message(network_core.EndPhaseOneMessage())
        self.server.broadcast_message(network_core.EndPhaseTwoMessage())
        self.logger.info("Everyone is disqualified, ending game...")

    def _end_game(self):
        self.server.broadcast_message(network_core.EndPhaseTwoMessage())
        self.logger.info("No more moves, ending game...")

    def _disqualify_player(self, player, err):
        self.history.add_disqualification(player)
        self.logger.info("Player {} Disqualified! {}".format(player.value, err))
        self.game.disqualify_player(player)
        self.server.broadcast_message(network_core.DisqualificationMessage(player))


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

    # Play actual game
    server = Server(board, 0, 1)
    server.start()
    server.join()

    # Save game history
    server.history.write('history.json')

    # Load game history
    history = serial.GameHistory()
    history.read('history.json')

    # Try replaying it
    game = GameState(board)
    for entry in history:
        game = game.execute_move(entry.player, entry.pos, entry.choice)
    print(game.board.map_string())

    # Try replaying the game from a specific part of the replay.
    # Note that we can replay with only one client connected.
    basic_server = network_core.BasicServer(board)
    basic_server.start()

    basic_server.accept_client()

    basic_server.set_player_for_group(0, Field.PLAYER_ONE)

    for i in range(0, 8):
        move = history[i]
        basic_server.broadcast_message(network_core.MoveNotificationMessage(move.pos, move.choice, move.player))

    # Now give the AI some more time for a specific move
    basic_server.send_player_message(Field.PLAYER_ONE, network_core.MoveRequestMessage(20_000, 0))

    basic_server.stop()
