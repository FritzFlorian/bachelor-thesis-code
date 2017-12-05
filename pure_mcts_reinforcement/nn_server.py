"""Wrapper to run a Neural Network in a separate process.

All interaction with the neural network is done using zeromq communication.
The NNServer has to be defined in an separate file as we only want to
use it from cpython, as pypy will not allow us to even import tensorflow."""
import tensorflow as tf
import zmq
import time
import pickle
import pure_mcts_reinforcement.nn_client as nn_client
import pure_mcts_reinforcement.util as util
import logging


class NeuralNetworkServer:
    def __init__(self, port, neural_network, input_conversion, output_conversion, batch_size):
        self.port = port
        self.neural_network = neural_network
        self.input_conversion = input_conversion
        self.output_conversion = output_conversion
        self.stopped = False
        self.socket = None
        self.execution_responses = []
        self.batch_size = batch_size

    def run(self):
        # Init network code. Router is used because we want to synchronize
        # all request in in a request-reply fashion.
        context = zmq.Context()
        self.socket = context.socket(zmq.ROUTER)
        self.socket.bind('tcp://*:{}'.format(self.port))

        # Setup a tensorflow session to be used for the whole run.
        graph = tf.Graph()
        with graph.as_default():
            self.neural_network.construct_network()
            with tf.Session() as sess:
                self.neural_network.init_network()

                while not self.stopped:
                    try:
                        # Multipart messages are needed to correctly map the response
                        message = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                        response_ids = message[0:-2]
                        message_content = pickle.loads(message[-1])

                        # Process the incoming message...
                        if isinstance(message_content, nn_client.ShutdownRequest):
                            self._process_shutdown_request(response_ids, message_content)
                        elif isinstance(message_content, nn_client.ExecutionRequest):
                            self._process_execution_request(response_ids, message_content, sess)
                        elif isinstance(message_content, nn_client.TrainingRequest):
                            self._process_training_request(response_ids, message_content, sess)
                        elif isinstance(message_content, nn_client.SaveWeightsResponse):
                            self._process_save_weights_request(response_ids, message_content, sess)
                        elif isinstance(message_content, nn_client.LoadWeightsRequest, sess):
                            self._process_load_weights_request(response_ids, message_content, sess)
                        elif isinstance(message_content, nn_client.ConversionFunctionRequest):
                            self._process_conversion_function_request(response_ids, message_content)
                        else:
                            logging.error("Unknown message '{}' received!".format(message_content))

                    except zmq.ZMQError:
                        # Also execute not full batches if no new data arrived in time
                        if len(self.execution_responses) >= 1:
                            self._execute_batch(sess)
                        else:
                            # Don't  busy wait all the time
                            time.sleep(0.01)

        self.socket.close()
        context.term()

    def _process_execution_request(self, response_ids, message_content, sess):
        response = nn_client.ExecutionResponse(response_ids, message_content.input_array)
        self.execution_responses.append(response)

        if len(self.execution_responses) > self.batch_size:
            self._execute_batch(sess)

    def _execute_batch(self, sess):
        inputs = [execution_response.input_array for execution_response in self.execution_responses]
        outputs = self.neural_network.execute_batch(sess, inputs)

        for i in range(len(outputs)):
            self.execution_responses[i].set_output(outputs[i])
            self.socket.send_multipart(self.execution_responses[i].to_multipart())

        self.execution_responses = []

    def _process_training_request(self, response_ids, message_content, sess):
        self.neural_network.train_batch(sess, message_content.input_arrays, message_content.target_arrays)

        response = nn_client.Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_save_weights_request(self, response_ids, message_content, sess):
        checkpoint_content = util.save_neural_net_to_zip_binary(self.neural_network, sess)
        response = nn_client.SaveWeightsResponse(response_ids, checkpoint_content)
        self.socket.send_multipart(response.to_multipart())

    def _process_load_weights_request(self, response_ids, message_content, sess):
        util.load_neural_net_from_zip_binary(message_content.checkpoint_content, self.neural_network, sess)

        response = nn_client.Response(response_ids)
        self.socket.send_multipart(response.to_multipart())

    def _process_conversion_function_request(self, response_ids, message_content):
        response = nn_client.ConversionFunctionResponse(response_ids, self.input_conversion, self.output_conversion)
        self.socket.send_multipart(response.to_multipart())

    def _process_shutdown_request(self, response_ids, message_content):
        self.stopped = True

        response = nn_client.Response(response_ids)
        self.socket.send_multipart(response)
