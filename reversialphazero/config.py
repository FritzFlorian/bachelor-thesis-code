import hometrainer.config
import reversialphazero.ai_trivial_agent
import definitions


class CustomConfiguration(hometrainer.config.Configuration):
    def __init__(self):
        super().__init__()
        # Overwrite all defaults to the defaults of these experiments

        # Default Settings for Search/Training
        # Number of game states for one training batch
        self._training_batch_size = 64
        # Number of last games used for training
        self._training_history_size = 128
        # Simulations per selfplay/selfeval turn
        self._simulations_per_turn = 128
        # Turn time for each player during ai evaluation
        self._external_evaluation_turn_time = 1.0

        # Number of selfplay games for each iteration
        self._n_self_play = 42
        # Number of self evaluation games for each iteration, 0 will skip it
        self._n_self_eval = 21
        # Number of evaluation games against the ai-trivial client for each client, 0 will skip it
        self._n_external_eval = 14

        # The self evaluation avg. score needed to see this iteration as new best.
        # This means if the new weights scored >= this value they will be chosen as best weights.
        # 0.05 is a sane default for scores between -1 and 1.
        self._needed_avg_self_eval_score = 0.05

        # C-PUCT used in Tree search
        self._c_puct = 3

        # Configure the number of concurrent search threads, 1 means no multithreading
        self._n_search_threads_external_eval = 4
        self._n_search_threads_self_eval = 4
        self._n_search_threads_selfplay = 4

    def external_evaluation_possible(self):
        return definitions.AI_TRIVIAL_AVAILABLE

    def external_ai_agent(self, start_game_state):
        return reversialphazero.ai_trivial_agent.AITrivialAgent()

    def zmq_use_secure_connection(self):
        return True

    def zmq_client_secret(self):
        return definitions.CLIENT_SECRET

    def zmq_server_secret(self):
        return definitions.SERVER_SECRET

    def zmq_server_public(self):
        return definitions.SERVER_PUBLIC

    def zmq_public_keys_dir(self):
        return definitions.PUBLIC_KEYS_DIR
