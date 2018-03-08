import pyximport; pyximport.install()
import unittest
import socket
import threading
import reversi.network as network
from reversi.game_core import Direction, Field, Board


class TestNetworkCommunication(unittest.TestCase):
    def setUp(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(('localhost', 0))
        self.server.listen(1)

        port = self.server.getsockname()[1]

        def connect_client():
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect(('localhost', port))
        threading.Thread(target=connect_client).start()

        self.server_conn, _ = self.server.accept()

    def tearDown(self):
        self.server_conn.close()
        self.client.close()
        self.server.close()

    def test_group_message(self):
        group_message = network.GroupNumberMessage(group_number=1)
        group_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual(1, answer.group_number)

    def test_board_message(self):
        board = Board("""\
        2
        1
        2 3
        3 3
        0 0 0
        1 0 2
        0 0 0
        0 0 0 <-> 2 0 2
        """)

        board_message = network.BoardMessage(board)
        board_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual(2, answer.board.n_players)
        self.assertEqual(1, answer.board.n_overwrite)
        self.assertEqual(Field.PLAYER_ONE, answer.board[(0, 1)])
        self.assertEqual(((2, 0), Direction.LEFT), answer.board.transitions[((0, 0), Direction.TOP)])

    def test_player_number_message(self):
        player_message = network.PlayerNumberMessage(player=Field.PLAYER_ONE)
        player_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual(Field.PLAYER_ONE, answer.player)

    def test_move_request_message(self):
        move_request_message = network.MoveRequestMessage(time_limit=100, depth_limit=5)
        move_request_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual(100, answer.time_limit)
        self.assertEqual(5, answer.depth_limit)

    def test_move_response_message(self):
        move_response_message = network.MoveResponseMessage((5, 10), Field.PLAYER_FOUR)
        move_response_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual((5, 10), answer.pos)
        self.assertEqual(Field.PLAYER_FOUR, answer.choice)

    def test_move_notification_test(self):
        move_notification_message = network.MoveNotificationMessage((5, 10), None, Field.PLAYER_TWO)
        move_notification_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual((5, 10), answer.pos)
        self.assertEqual(None, answer.choice)
        self.assertEqual(Field.PLAYER_TWO, answer.player)

    def test_disqualification_message(self):
        disqualification_message = network.DisqualificationMessage(Field.PLAYER_ONE)
        disqualification_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertEqual(Field.PLAYER_ONE, answer.player)

    def test_end_phase_one_message(self):
        end_message = network.EndPhaseOneMessage()
        end_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertTrue(isinstance(answer, network.EndPhaseOneMessage))

    def test_end_phase_two_message(self):
        end_message = network.EndPhaseTwoMessage()
        end_message.write_to_conn(self.server_conn)
        answer = network.read_message_from_conn(self.client)

        self.assertTrue(isinstance(answer, network.EndPhaseTwoMessage))


class TestBasicClientServer(unittest.TestCase):
    def test_typical_interaction(self):
        board = Board("""\
        2
        0
        0 0
        2 2
        0 0
        0 0
        """)

        server = network.BasicServer(board)
        client = network.BasicClient(14)
        server.start()

        def connect_client():
            client.start()
        t = threading.Thread(target=connect_client)
        t.start()

        self.assertEqual(14, server.accept_client())
        self.assertTrue(server.clients_by_group[14])

        server.set_player_for_group(14, Field.PLAYER_ONE)

        t.join()
        self.assertEqual(Field.PLAYER_ONE, client.player)

        server.send_player_message(Field.PLAYER_ONE, network.MoveRequestMessage(1000, 0))
        rec = client.read_message()
        self.assertEqual(1000, rec.time_limit)
        self.assertEqual(0, rec.depth_limit)

        client.send_message(network.MoveResponseMessage((1, 2), None))
        rec = server.read_player_message(Field.PLAYER_ONE, network.MoveResponseMessage)
        self.assertEqual((1, 2), rec.pos)
        self.assertEqual(None, rec.choice)

        client.stop()
        server.stop()

if __name__ == '__main__':
    unittest.main()
