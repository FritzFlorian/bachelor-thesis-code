"""Network Code for ReversiXT.

This includes package encoding/decoding and all primitives,
but excludes actual logic for a full client/server."""
from reversi.game_core import Board, Field


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

def read_message_from_conn(conn):
    message_type = conn.recv(1)

    if message_type == 1:
        pass
    elif message_type == 2:
        pass
        # ...


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
        write_8_bit_int(1)
        write_32_bit_int(8)

        write_8_bit_int(conn, self.group_number)


class BoardMessage(Message):
    def __init__(self, board):
        super().__init__()
        self.board = board

    def read_from_conn(self, conn):
        self.read_message_length(conn)
        board_string = read_string(conn, self.message_length)
        self.board = Board(board_string)

    def write_to_conn(self, conn):
        write_8_bit_int(2)

        board_string = self.board.__string__()
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
        write_8_bit_int(3)
        write_32_bit_int(8)

        write_8_bit_int(self.player_number)


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
        write_8_bit_int(4)
        write_32_bit_int(32 + 8)

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
        self.choice = read_8_bit_int(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(5)
        write_32_bit_int(16 + 16 + 8)

        x, y = self.pos
        write_16_bit_int(conn, x)
        write_16_bit_int(conn, y)
        write_8_bit_int(conn, self.choice)


class MoveNotificationMessage(Message):
    def __init__(self, pos=None, choice=None, player=None):
        super().__init__()
        self.pos = pos
        self.choice = choice
        self.player = player

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.pos = read_16_bit_int(conn), read_16_bit_int(conn)
        self.choice = read_8_bit_int(conn)
        self.player = Field(chr(read_8_bit_int(conn) + ord(Field.PLAYER_ONE.value)))

    def write_to_conn(self, conn):
        write_8_bit_int(6)
        write_32_bit_int(16 + 16 + 8 + 8)

        x, y = self.pos
        write_16_bit_int(conn, x)
        write_16_bit_int(conn, y)
        write_8_bit_int(conn, self.choice)
        write_8_bit_int(conn, ord(self.player))


class DisqualificationMessage(Message):
    def __init__(self, player=None):
        super().__init__()
        self.player = player

    def read_from_conn(self, conn):
        self.read_message_length(conn)

        self.player = Field(chr(read_8_bit_int(conn) + ord(Field.PLAYER_ONE.value)))

    def write_to_conn(self, conn):
        write_8_bit_int(7)
        write_32_bit_int(8)

        write_8_bit_int(conn, ord(self.player))


class EndPhaseOneMessage(Message):
    def __init__(self):
        super().__init__()

    def read_from_conn(self, conn):
        self.read_message_length(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(8)
        write_8_bit_int(0)


class EndPhaseTwoMessage(Message):
    def __init__(self):
        super().__init__()

    def read_from_conn(self, conn):
        self.read_message_length(conn)

    def write_to_conn(self, conn):
        write_8_bit_int(9)
        write_8_bit_int(0)
