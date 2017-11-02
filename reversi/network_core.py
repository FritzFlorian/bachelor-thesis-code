"""Network Code for ReversiXT.

This includes package encoding/decoding and all primitives,
but excludes actual logic for a full client/server.

The BasicClient and BasicServer classes wrap all functions needed
to run a client/server, but do not include code for server specific control flow.
They should be used to build up the actual client/server as well as
altered implementations useful for gathering data."""
from reversi.game_core import Board, Field, PLAYERS
import socket
import logging


DEFAULT_PORT = 7777
DEFAULT_HOST = 'localhost'
# Always timeout moves if they take longer than 5 Minutes, no matter the depth limit
MOVE_TIMEOUT = 5 * 60
# Clients must answer basic request (not moves) in at most 10 seconds
GENERAL_TIMEOUT = 10


class DisqualifiedError(Exception):
    """All 'known' client errors and timeouts lead to an disqualified exception for that player."""
    def __init__(self, group, player, message, cause=None):
        self.group = group
        self.player = player
        self.message = message
        self.cause = cause


class BasicServer:
    """Basic ReversiXT Network Server. Use this to build your custom Server.

    Basic Usage:
    >>> server = BasicServer(4242)
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
    def __init__(self, port=DEFAULT_PORT):
        self.logger = logging.getLogger("BasicServer ({})".format(port))

        self.port = port
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.clients_by_group = dict()
        self.clients_by_player = dict()

        self.player_by_client = dict()
        self.group_by_client = dict()

        self.next_group = 0

    def start(self):
        self.logger.info("Starting server on port {}...".format(self.port))
        self.server.bind((socket.gethostname(), self.port))

    def stop(self):
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
        self._send_message(client, PlayerNumberMessage(player))

        self.clients_by_player[player] = client
        self.player_by_client[client] = player

    def read_player_message(self, player, message_class, timeout=MOVE_TIMEOUT):
        timeout = max(timeout, MOVE_TIMEOUT)
        client = self.clients_by_player[player]

        return self._read_message(client, message_class, timeout)

    def send_player_message(self, player, message):
        client = self.clients_by_player[player]
        self._send_message(client, message)

    def _read_message(self, client, message_class, timeout=GENERAL_TIMEOUT):
        group = self.group_by_client[client]
        player = None
        if client in self.player_by_client:
            player = self.player_by_client[client]

        try:
            client.settimeout(timeout)
            message = read_message_from_conn(client)
            if isinstance(message, message_class):
                return message

            # Handle most cases of client errors
            self.logger.error("User sent wrong message type ({})! Group: {}, Player: {}".format(client, group, player))
            raise DisqualifiedError(group, player, "Client set wrong message type!")
        except socket.timeout as err:
            self.logger.error("Detected Timeout ({})! Group: {}, Player: {}".format(client, group, player))
            raise DisqualifiedError(group, player, "Client Timeout!", err)
        except socket.error as err:
            self.logger.error("Detected Network Error ({})! Group: {}, Player: {}".format(client, group, player))
            raise DisqualifiedError(group, player, "Network Error!", err)

    def _send_message(self, client, message):
        try:
            message.write_to_conn(client)
        except socket.error as err:
            group = self.group_by_client[client]
            player = None
            if client in self.player_by_client:
                player = self.player_by_client[client]
            self.logger.error("Detected Network Error ({})! Group: {}, Player: {}".format(client, group, player))
            raise DisqualifiedError(group, player, "Network Error!", err)


class BasicClient:
    """Basic ReversiXT Network Client. Use this to build your custom Client.

    Basic Usage:
    >>> client = BasicClient(14, 'localhost', 4242)
    >>> client.start()
    >>>
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

    def start(self):
        """Connects to the server and exchanges group and player number."""
        self.logger.info("Connecting to server...")
        self.client.connect((self.host, self.port))

        group_message = GroupNumberMessage(self.group)
        self.send_message(group_message)
        self.logger.info("Connected as group {}, waiting for player number...")

        self.player = self.read_message().player_number
        self.logger.info("Client was assigned player {}.".format(self.player.value))

    def stop(self):
        self.client.close()

    def send_message(self, message):
        message.write_to_conn(self.client)

    def read_message(self):
        return read_message_from_conn(self.client)


def read_n_bytes(conn, n_bytes):
    toread = n_bytes

    buf = bytearray(toread)
    view = memoryview(buf)
    while toread:
        nbytes = conn.recv_into(view, toread)
        view = view[nbytes:]  # slicing views is cheap
        toread -= nbytes

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
        return ord(choice.value) - ord(Field.PLAYER_ONE.value) + 1
    if choice == 'bomb':
        return 20
    if choice == 'overwrite':
        return 21

    return 0


def integer_to_choice(integer):
    if 1 <= integer <= 8:
        return Field(chr(integer - 1 + ord(Field.PLAYER_ONE.value)))
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
        write_32_bit_int(conn, 8)

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
    def __init__(self, player_number=None):
        super().__init__()
        self.player_number = player_number

    def read_from_conn(self, conn):
        self.read_message_length(conn)
        self.player_number = read_8_bit_int(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 3)
        write_32_bit_int(conn, 8)

        write_8_bit_int(conn, self.player_number)


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
        write_32_bit_int(conn, 32 + 8)

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
        write_32_bit_int(conn, 16 + 16 + 8)

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
        self.player = Field(chr(read_8_bit_int(conn) + ord(Field.PLAYER_ONE.value) - 1))

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 6)
        write_32_bit_int(conn, 16 + 16 + 8 + 8)

        x, y = self.pos
        write_16_bit_int(conn, x)
        write_16_bit_int(conn, y)
        write_8_bit_int(conn, choice_to_integer(self.choice))
        write_8_bit_int(conn, ord(self.player.value) - ord(Field.PLAYER_ONE.value) + 1)


class DisqualificationMessage(Message):
    def __init__(self, player=None):
        super().__init__()
        self.player = player

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.player = Field(chr(read_8_bit_int(conn) + ord(Field.PLAYER_ONE.value) - 1))

    def write_to_conn(self, conn):
        write_8_bit_int(conn, 7)
        write_32_bit_int(conn, 8)

        write_8_bit_int(conn, ord(self.player.value) - ord(Field.PLAYER_ONE.value) + 1)


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
