# Certificates for network communication

The project uses pyzmq for it's network communication components.
It sends objects using the build in pickle module. Without proper security
on the network components attackers could potentially execute any python code on
a victims system, therefore keys are required to secure the communication.


Simply run `generate_certificates.py` to generate a new set of keys.
Copy the `private_keys` and `public_keys` onto all systems participating
in the training.


The script for generating the keys and other information can be found here:
https://github.com/zeromq/pyzmq/tree/master/examples/security
