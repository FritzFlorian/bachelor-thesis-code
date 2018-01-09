import tempfile
import os
import shutil
import zipfile
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator
import definitions


def save_neural_net_to_zip_binary(neural_network, session):
    with tempfile.TemporaryDirectory() as base_dir:
        # Prepare Saving Path's
        checkpoint_dir = os.path.join(base_dir, 'checkpoint')
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
        os.makedirs(checkpoint_dir)

        # Save it
        neural_network.save_weights(session, checkpoint_file)

        # Pack it up to send it through the network
        checkpoint_zip = os.path.join(base_dir, 'checkpoint')
        shutil.make_archive(checkpoint_zip, 'zip', checkpoint_dir)
        with open(checkpoint_zip + '.zip', 'rb') as file:
            return file.read()


def load_neural_net_from_zip_binary(zip_binary, neural_network, session):
    with tempfile.TemporaryDirectory() as base_dir:
        # Prepare Saving Path's
        checkpoint_dir = os.path.join(base_dir, 'checkpoint')
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
        checkpoint_zip = os.path.join(base_dir, 'checkpoint.zip')
        os.makedirs(checkpoint_dir)

        # Prepare the checkpoint data for loading
        with open(checkpoint_zip, 'wb') as file:
            file.write(zip_binary)

        with zipfile.ZipFile(checkpoint_zip, "r") as zip_ref:
            zip_ref.extractall(checkpoint_dir)

        neural_network.load_weights(session, checkpoint_file)


def secure_client_connection(client, ctx, only_localhost=False):
    """Configures certificates for zeromq client connection."""
    _init_auth(ctx, only_localhost)

    client_public, client_secret = zmq.auth.load_certificate(definitions.CLIENT_SECRET)
    client.curve_secretkey = client_secret
    client.curve_publickey = client_public

    server_public, _ = zmq.auth.load_certificate(definitions.SERVER_PUBLIC)
    # The client must know the server's public key to make a CURVE connection.
    client.curve_serverkey = server_public


def secure_server_connection(server, ctx, only_localhost=False):
    """Configures certificates for zeromq server connection."""
    _init_auth(ctx, only_localhost)

    server_public, server_secret = zmq.auth.load_certificate(definitions.SERVER_SECRET)
    server.curve_secretkey = server_secret
    server.curve_publickey = server_public
    server.curve_server = True  # must come before bind


def _init_auth(ctx, only_localhost):
    auth = ThreadAuthenticator(ctx)
    auth.start()

    if only_localhost:
        auth.allow('127.0.0.1')
    else:
        auth.allow('*')

    auth.configure_curve(domain='*', location=definitions.PUBLIC_KEYS_DIR)

def count_files(dir_path):
    return len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
