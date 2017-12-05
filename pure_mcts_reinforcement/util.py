import tempfile
import os
import shutil
import zipfile


def save_neural_net_to_zip_binary(neural_network, session):
    with tempfile.TemporaryDirectory() as base_dir:
        # Prepare Saving Path's
        checkpoint_dir = os.path.join(base_dir, 'checkpoint')
        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.ckpt')
        os.makedirs(checkpoint_dir)

        # Save it
        neural_network.save_weights(session, checkpoint_file)

        # Pack it up to send it through the network
        checkpoint_zip = os.path.join(base_dir, 'checkpoint.zip')
        shutil.make_archive(checkpoint_zip, 'zip', checkpoint_dir)
        with open(checkpoint_zip, 'rb') as file:
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
