import subprocess
import time
import os
import re

# Path to the reversi_no_gl server binary
# Set this for running on own machine
server_path = os.path.join(os.getcwd(), "server_nogl")

# Path to the reversi client binary
# Set this for running on own machine
client_path = os.path.join(os.getcwd(), "ai_trivial")

# General Settings
logs_path = "raw_logs"
map_path = "standard.map"
num_players = 2
depths = range(1, 10)

def run_match(depth):
    server = subprocess.Popen([server_path, "-m", map_path, "-d", str(depth)], stdout=subprocess.PIPE, universal_newlines=True)
    time.sleep(1)
    for i in range(num_players):
        subprocess.Popen([client_path])

    logfile = os.path.join(os.getcwd(), logs_path, "depth-{0}.log".format(depth))
    directory = os.path.dirname(logfile)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(logfile, 'w') as f:
        asci_escape_colors = re.compile(r'\x1b[^m]*m')
        
        raw_log_output = server.communicate()[0]
        stripped_log_output = asci_escape_colors.sub('', raw_log_output)
        
        f.write(stripped_log_output)

if __name__ == "__main__":
    for d in depths:
        run_match(d)
