"""Network Code for ReversiXT.

This includes package encoding/decoding and all primitives,
but excludes actual logic for a full client/server.

The BasicClient and BasicServer classes wrap all functions needed
to run a client/server, but do not include code for server specific control flow.
They should be used to build up the actual client/server as well as
altered implementations useful for gathering data."""
from reversi.game_core import GameState, Board, Field, DisqualifiedError, PLAYERS
import socket
import logging
import time
import definitions
import threading
import random


DEFAULT_PORT = definitions.REVERSI_MATCH_SERVER_DEFAULT_PORT
DEFAULT_HOST = definitions.REVERSI_MATCH_SERVER_DEFAULT_HOST
# Always timeout moves if they take longer than 5 Minutes, no matter the depth limit
MOVE_TIMEOUT = 5 * 60
# Clients must answer basic request (not moves) in at most 10 seconds
GENERAL_TIMEOUT = 10


class BasicServer:
    """Basic ReversiXT Network Server. Use this to build your custom Server.

    Basic Usage:
    >>> server = BasicServer(board, 4242)
    >>> server.start()
    >>>
    >>> one = server.accept_client()
    >>> two = server.accept_client()
    >>>
    >>> server.set_player_for_group(one, Field.PLAYER_ONE)
    >>> server.set_player_for_group(two, Field.PLAYER_TWO)
    >>>
    >>> server.send_player_message(Field.PLAYER_ONE, MoveRequestMessage(...))
    >>> response = server.read_player_message(Field.PLAYER_ONE, MoveResponseMessage, 10)
    >>>
    >>> for p in [Field.PLAYER_ONE, Field.PLAYER_TWO]:
    >>>     server.send_player_message(p, MoveNotificationMessage(...))
    >>>
    >>> server.stop()
    """
    def __init__(self, board, port=DEFAULT_PORT):
        self.logger = logging.getLogger("BasicServer ({})".format(port))

        self.board = board
        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.clients_by_group = dict()
        self.clients_by_player = dict()

        self.player_by_client = dict()
        self.group_by_client = dict()

        self.next_group = 0

    def start(self):
        self.logger.info("Starting server on port {}...".format(self.port))
        self.server.bind(('0.0.0.0', self.port))
        self.server.listen(1)

    def stop(self):
        for k, v in self.clients_by_group.items():
            v.close()
        self.server.close()

    def accept_client(self):
        """Waits for a client to connect and send its group number. Returns the group number."""
        self.logger.info("Waiting for client connection...")
        client, address = self.server.accept()

        # Read the group, simply assign one if there are conflicts
        group = self._read_message(client, GroupNumberMessage).group_number
        while group in self.clients_by_group:
            group = self.next_group
            self.next_group = self.next_group + 1

        self.clients_by_group[group] = client
        self.group_by_client[client] = group

        self.logger.info("Client from group {} connected ({}).".format(group, address))
        return group

    def set_player_for_group(self, group, player):
        client = self.clients_by_group[group]

        self._send_message(client, BoardMessage(self.board))
        self._send_message(client, PlayerNumberMessage(player))

        self.clients_by_player[player] = client
        self.player_by_client[client] = player

    def read_player_message(self, player, message_class, timeout=MOVE_TIMEOUT):
        if timeout == 0:
            timeout = MOVE_TIMEOUT
        client = self.clients_by_player[player]

        return self._read_message(client, message_class, timeout)

    def send_player_message(self, player, message):
        client = self.clients_by_player[player]
        self._send_message(client, message)

    def broadcast_message(self, message):
        for k, v in self.clients_by_group.items():
            try:
                self._send_message(v, message)
            except DisqualifiedError:
                # Ignore these errors at this point, they do not influence the further game.
                # We will disqualify players only if they don't send us messages in time.
                pass

    def _read_message(self, client, message_class, timeout=GENERAL_TIMEOUT):
        player = self.player_by_client.get(client, None)

        try:
            client.settimeout(float(timeout))
            client.setblocking(True)
            message = read_message_from_conn(client)
            if isinstance(message, message_class):
                return message

            # Handle most cases of client errors
            raise DisqualifiedError(player, "Client sent wrong message type!")
        except socket.timeout as err:
            raise DisqualifiedError(player, "Client Timeout!", err)
        except socket.error as err:
            raise DisqualifiedError(player, "Network Error!", err)

    def _send_message(self, client, message):
        try:
            message.write_to_conn(client)
        except socket.error as err:
            player = self.player_by_client.get(client, None)

            raise DisqualifiedError(player, "Network Error!", err)


class BasicClient:
    """Basic ReversiXT Network Client. Use this to build your custom Client.

    Basic Usage:
    >>> client = BasicClient(14, 'localhost', 4242)
    >>> client.start()
    >>>
    >>> print(client.board)
    >>> print(client.player)
    >>>
    >>> message = client.read_message()
    >>> if isinstance(message, MoveRequestMessage):
    >>>     client.send_message(MoveResponseMessage(...))
    >>> elif isinstance(message, DisqualificationMessage):
    >>>     ...
    """
    def __init__(self, group, host=DEFAULT_HOST, port=DEFAULT_PORT):
        self.logger = logging.getLogger("BasicClient ({}:{})".format(host, port))

        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = port
        self.group = group
        self.player = None
        self.board = None

    def start(self):
        """Connects to the server and exchanges group and player number."""
        self.logger.info("Connecting to server...")
        self.client.connect((self.host, self.port))

        group_message = GroupNumberMessage(self.group)
        self.send_message(group_message)
        self.logger.info("Connected as group {}, waiting for board...".format(self.group))

        self.board = self.read_message().board
        self.logger.info("Board received, waiting for player number...")

        self.player = self.read_message().player
        self.logger.info("Client was assigned player {}.".format(self.player))

    def stop(self):
        self.client.close()

    def send_message(self, message):
        message.write_to_conn(self.client)

    def read_message(self):
        return read_message_from_conn(self.client)


class Server(threading.Thread):
    def __init__(self, board, time, depth, port=DEFAULT_PORT, group_to_player=None):
        super().__init__()
        self.logger = logging.getLogger("Server ({})".format(port))
        self.game = GameState(board)
        self.server = BasicServer(board, port)
        self.time = time * 1000
        self.depth = depth
        self.group_to_player = group_to_player

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
        if self.group_to_player:
            for g, p in self.group_to_player.items():
                self.server.set_player_for_group(g, p)
        else:
            for i in range(self.game.board.n_players):
                self.server.set_player_for_group(groups[i], Field.PLAYER_ONE+ i)

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
                self.server.broadcast_message(EndPhaseOneMessage())

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
        move_request = MoveRequestMessage(self.times[player], self.depth)
        self.server.send_player_message(player, move_request)

    def _process_move_answer(self, player):
        move_response = \
            self.server.read_player_message(player, MoveResponseMessage, self.times[player]/1000)
        self.logger.info("Player Move: ({}, {})".format(move_response.pos, move_response.choice))

        self.game = self.game.execute_move(player, move_response.pos, move_response.choice)
        if not self.game:
            raise DisqualifiedError("Client send invalid move!", player)

        self.logger.info(self.game.board.board_string())

    def _broadcast_last_move_notification(self):
        (player, pos, choice) = self.game.last_move
        move_notification = MoveNotificationMessage(pos, choice, player)
        self.server.broadcast_message(move_notification)

    def _inc_player_time(self, player):
        if self.times[player] < 0:
            self.times[player] = 0
        self.times[player] = self.times[player] + self.time

    def _end_game_disqualified(self):
        if not self.game.bomb_phase:
            self.server.broadcast_message(EndPhaseOneMessage())
        self.server.broadcast_message(EndPhaseTwoMessage())
        self.logger.info("Everyone is disqualified, ending game...")

    def _end_game(self):
        self.server.broadcast_message(EndPhaseTwoMessage())
        self.logger.info("No more moves, ending game...")

    def _disqualify_player(self, player, err):
        self.logger.info("Player {} Disqualified! {}".format(player, err))
        self.game.disqualify_player(player)
        self.server.broadcast_message(DisqualificationMessage(player))


def run_server_example():
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


class Client(threading.Thread):
    def __init__(self, group, find_move, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 game_start_callback=None, game_end_callback=None, move_callback=None):
        super().__init__()
        self.logger = logging.getLogger("Client ({})".format(group))
        self.client = BasicClient(group, host, port)

        self.find_move = find_move
        self.game_start_callback = game_start_callback
        self.game_end_callback = game_end_callback
        self.move_callback = move_callback

        self.game_state = None

    def run(self):
        self.client.start()
        self.game_state = GameState(self.client.board)

        if self.game_start_callback:
            self.game_start_callback(self.game_state, self.client.player)

        self._game_loop()
        self.logger.info("Game Ended")

        if self.game_end_callback:
            self.game_end_callback(self.game_state)

        self.client.stop()

    def _game_loop(self):
        while True:
            message = self.client.read_message()
            if isinstance(message, EndPhaseTwoMessage):
                return
            elif isinstance(message, EndPhaseOneMessage):
                self.game_state.bomb_phase = True
                self.logger.info("Phase One Ended")
            elif isinstance(message, MoveRequestMessage):
                self.logger.info("Move Request from server ({}, {})".format(message.time_limit, message.depth_limit))
                (player, pos, choice) = self.find_move(self.game_state, message.time_limit, message.depth_limit)
                self.logger.info("Answer: {}, {}".format(pos, choice))
                move_message = MoveResponseMessage(pos, choice)
                self.client.send_message(move_message)
            elif isinstance(message, DisqualificationMessage):
                self.logger.info("Player {} Disqualified!".format(message.player))
                self.game_state.disqualify_player(message.player)
                if message.player == self.client.player:
                    self.logger.info("Client was disqualified, shutting down...")
                    return
            elif isinstance(message, MoveNotificationMessage):
                old_game_state = self.game_state
                self.game_state = self.game_state.execute_move(message.player, message.pos, message.choice)

                if self.move_callback:
                    self.move_callback(old_game_state, (message.player, message.pos, message.choice), self.game_state)

def run_client_example():
    logging.basicConfig(level=logging.INFO)

    def find_move(game, time_limit, depth_limit):
        possible = game.get_next_possible_moves()
        return random.choice(possible).last_move

    client = Client(14, find_move)
    client.start()
    client.join()


def read_n_bytes(conn, n_bytes):
    toread = n_bytes
    start_time = time.time()

    buf = bytearray(toread)
    view = memoryview(buf)
    while toread:
        nbytes = conn.recv_into(view, toread)
        view = view[nbytes:]  # slicing views is cheap
        toread -= nbytes

        # Stop receiving after the players time ran out
        if conn.timeout and 0 < conn.timeout < time.time() - start_time:
            raise socket.timeout()

    return buf


def read_8_bit_int(conn):
    data = read_n_bytes(conn, 1)
    return int.from_bytes(data, byteorder='big', signed=False)


def read_16_bit_int(conn):
    data = read_n_bytes(conn, 2)
    return int.from_bytes(data, byteorder='big', signed=False)


def read_32_bit_int(conn):
    data = read_n_bytes(conn, 4)
    return int.from_bytes(data, byteorder='big', signed=False)


def write_8_bit_int(conn, i):
    data = i.to_bytes(1, byteorder='big', signed=False)
    conn.send(data)


def write_16_bit_int(conn, i):
    data = i.to_bytes(2, byteorder='big', signed=False)
    conn.send(data)


def write_32_bit_int(conn, i):
    data = i.to_bytes(4, byteorder='big', signed=False)
    conn.send(data)


def read_string(conn, length):
    data = read_n_bytes(conn, length)
    return data.decode("utf-8")


def choice_to_integer(choice):
    if choice in PLAYERS:
        return Field.to_player_int(choice)
    if choice == 'bomb':
        return 20
    if choice == 'overwrite':
        return 21

    return 0


def integer_to_choice(integer):
    if 1 <= integer <= 8:
        return Field.int_to_player(integer)
    if integer == 20:
        return 'bomb'
    if integer == 21:
        return 'overwrite'

    return None


def read_message_from_conn(conn):
    message_type = read_8_bit_int(conn)

    message = None
    if message_type == 1:
        message = GroupNumberMessage()
    elif message_type == 2:
        message = BoardMessage()
    elif message_type == 3:
        message = PlayerNumberMessage()
    elif message_type == 4:
        message = MoveRequestMessage()
    elif message_type == 5:
        message = MoveResponseMessage()
    elif message_type == 6:
        message = MoveNotificationMessage()
    elif message_type == 7:
        message = DisqualificationMessage()
    elif message_type == 8:
        message = EndPhaseOneMessage()
    elif message_type == 9:
        message = EndPhaseTwoMessage()

    message.read_from_conn(conn)
    return message


class Message:
    def __init__(self):
        self.message_length = -1

    def read_from_conn(self, conn):
        raise NotImplementedError('Method must be overwritten by child classes!')

    def write_to_conn(self, conn):
        raise NotImplementedError('Method must be overwritten by child classes!')

    def read_message_length(self, conn):
        self.message_length = read_32_bit_int(conn)


class GroupNumberMessage(Message):
    def __init__(self, group_number=None):
        super().__init__()
        self.group_number = group_number

    def read_from_conn(self, conn):
        self.read_message_length(conn)
        self.group_number = read_8_bit_int(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 1)
        write_32_bit_int(conn, 1)

        write_8_bit_int(conn, self.group_number)


class BoardMessage(Message):
    def __init__(self, board=None):
        super().__init__()
        self.board = board

    def read_from_conn(self, conn):
        self.read_message_length(conn)
        board_string = read_string(conn, self.message_length)
        self.board = Board(board_string)

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 2)

        board_string = self.board.__string__().encode('utf-8')
        self.message_length = len(board_string)
        write_32_bit_int(conn, self.message_length)

        conn.send(board_string)


class PlayerNumberMessage(Message):
    def __init__(self, player=None):
        super().__init__()
        self.player = player

    def read_from_conn(self, conn):
        self.read_message_length(conn)
        self.player = Field.int_to_player(read_8_bit_int(conn))

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 3)
        write_32_bit_int(conn, 1)

        write_8_bit_int(conn, Field.to_player_int(self.player))


class MoveRequestMessage(Message):
    def __init__(self, time_limit=None, depth_limit=None):
        super().__init__()
        self.time_limit = time_limit
        self.depth_limit = depth_limit

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.time_limit = read_32_bit_int(conn)
        self.depth_limit = read_8_bit_int(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 4)
        write_32_bit_int(conn, 4 + 1)

        write_32_bit_int(conn, self.time_limit)
        write_8_bit_int(conn, self.depth_limit)


class MoveResponseMessage(Message):
    def __init__(self, pos=None, choice=None):
        super().__init__()
        self.pos = pos
        self.choice = choice

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.pos = read_16_bit_int(conn), read_16_bit_int(conn)
        self.choice = integer_to_choice(read_8_bit_int(conn))

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 5)
        write_32_bit_int(conn, 2 + 2 + 1)

        x, y = self.pos
        write_16_bit_int(conn, x)
        write_16_bit_int(conn, y)
        write_8_bit_int(conn, choice_to_integer(self.choice))


class MoveNotificationMessage(Message):
    def __init__(self, pos=None, choice=None, player=None):
        super().__init__()
        self.pos = pos
        self.choice = choice
        self.player = player

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.pos = read_16_bit_int(conn), read_16_bit_int(conn)
        self.choice = integer_to_choice(read_8_bit_int(conn))
        self.player = Field.int_to_player(read_8_bit_int(conn))

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 6)
        write_32_bit_int(conn, 2 + 2 + 1 + 1)

        x, y = self.pos
        write_16_bit_int(conn, x)
        write_16_bit_int(conn, y)
        write_8_bit_int(conn, choice_to_integer(self.choice))
        write_8_bit_int(conn, Field.to_player_int(self.player))


class DisqualificationMessage(Message):
    def __init__(self, player=None):
        super().__init__()
        self.player = player

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.player = Field.int_to_player(read_8_bit_int(conn))

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 7)
        write_32_bit_int(conn, 1)

        write_8_bit_int(conn, Field.to_player_int(self.player))


class EndPhaseOneMessage(Message):
    def __init__(self):
        super().__init__()

    def read_from_conn(self, conn):
        self.read_message_length(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 8)
        write_32_bit_int(conn, 0)


class EndPhaseTwoMessage(Message):
    def __init__(self):
        super().__init__()

    def read_from_conn(self, conn):
        self.read_message_length(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 9)
        write_32_bit_int(conn, 0)
