import hometrainer.agents
import random
import reversialphazero.core
import reversi.network as network
import reversi.tournament as tournament
import definitions


class AIJavaAgent(hometrainer.agents.Agent):
    def __init__(self,time_bonus=200):
        self.time_bonus = time_bonus

    """'Wrapper' to use the ai trivial binary as a agent.

    This sets up an ReversiXT server, starts an ai trivial and connects it to the server.
    The server then acts as a proxy between the ai trivial and the """
    def game_start(self, game_state, player):
        self.player = player

        # TODO: Handle Ports Properly
        found_port = False
        while not found_port:
            try:
                port = random.randint(2000, 4000)
                self.server = network.BasicServer(game_state.original_game_state.board, port)
                self.server.start()
                found_port = True
            except IOError:
                # TODO: Handle Ports Properly
                found_port = False
                print('Port Conflict, retry...')

        tournament.JavaClient(definitions.AI_JAVA_PATH).start('localhost', port)
        group = self.server.accept_client()
        self.server.set_player_for_group(group, player)

    def move_executed(self, old_game_state, move, new_game_state):
        # Execute the move
        player, pos, choice = move.internal_tuple
        self.server.broadcast_message(network.MoveNotificationMessage(pos, choice, player))

    def game_ended(self, game_state):
        self.server.broadcast_message(network.EndPhaseOneMessage())
        self.server.broadcast_message(network.EndPhaseTwoMessage())
        self.server.stop()

    def find_move_with_iteration_limit(self, game_state, move_iterations):
        self.server.send_player_message(self.player, network.MoveRequestMessage(0, move_iterations))
        move_response = self.server.read_player_message(self.player, network.MoveResponseMessage)
        selected_move = (self.player, move_response.pos, move_response.choice)

        return reversialphazero.core.ReversiMove(selected_move)

    def find_move_with_time_limit(self, game_state, move_time):
        # We give some extra time to the AI-Trivial, as it usually tries
        # to send a response about 200ms before timeout.
        self.server.send_player_message(self.player,
                                        network.MoveRequestMessage(round(move_time * 1000 + self.time_bonus), 0))
        move_response = self.server.read_player_message(self.player, network.MoveResponseMessage)
        selected_move = (self.player, move_response.pos, move_response.choice)

        return reversialphazero.core.ReversiMove(selected_move)
