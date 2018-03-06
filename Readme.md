# Bachelor Thesis - Python Code

This repository is a collection of the python code used during the bachelor
thesis of florian fritz.

## Installing/Running/Usage

### Installing
Follow the instructions to install all needed requirements to run the project.

- Install Python 3.5 64 bit & python3-pip
- Install virtualenv (via `pip install virtualenv')
- Be sure to have setup git correctly (https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
- Clone the project (https://github.com/FritzFlorian/bachelor-thesis-code/)
- Create a virtualenv for the project (cd project-folder; virtualenv -p python3 venv; source ./venv/bin/activate;)
    - WINDOWS NOTE:
        - Before activating the virtualenv for the first time run `Set-ExecutionPolicy RemoteSigned` in an administration shell
        - Then use `.\venv\Scripts\activate` to activate the virtualenv in any shell
- If running on windows
    - Install numpy and scipy manually if needed:
        - Pre compiled libraries: https://www.lfd.uci.edu/~gohlke/pythonlibs/
        - Choose the 64bit version of both libraries
        - Download the .whl files
        - run `pip install path/to/wheel.whl`
    - Install required C++ compiler for cython
        - This should work if you use the exact versions stated above (make sure to check Windows 8.1 SDK):
          http://landinghub.visualstudio.com/visual-cpp-build-tools
        - Or simply get the full VS Community (2015):
          https://www.visualstudio.com/vs/older-downloads/
        - More Details: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
        - If this does not work, add the visual studio 'rc.exe' to your PATH variable
- Run `pip install -r requirements.txt`
- If running on windows:
    - If installing all dependencies does not work, also install tensorflow, cython and pyzmq manually
- If running the training master with GPU acceleration
    - Install all needed CUDA drivers (as stated on the TF homepage, make sure to get the right ones),
      here are the current ones needed:
        - Graphics Drivers http://www.nvidia.de/Download/index.aspx
        - VS 2015 Community Edition: https://www.visualstudio.com/vs/older-downloads/
        - Nvidia Toolkit 8: https://developer.nvidia.com/cuda-80-ga2-download-archive
        - Nvidia cuDNN: https://developer.nvidia.com/rdp/cudnn-download
    - Do not forget to set your PATH (also stated on the TF homepage)
        - Usually for CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin`
        - And this: `C:\Users\YOURNAME\Documents\cudnn-8.0-windows10-x64-v6.0\cuda\bin`
    - `pip uninstall tensorflow`
    - `pip install tensorflow-gpu` (recommended version: 1.4.0)

### Updating the project
To update the project simply pull from the git repo. If needed recreate the virtualenv and reinstall the dependencies.

### Run the Project
The main experiments that are run during the bachelor thesis can be found in the subdirectories
`reinforcement/experiment_name`. To start training you will need to start one 'training_master' process
(best run on a machine with fast GPU) and at least one 'playing_slave'.
Most variables to run these are pre-configured in the separate experiments, to run them you normally only have
to provide the ip of the 'training_master'.

Before performing any of the below task follow these steps:
- cd projectdirectory
- on Unix
    - source ./venv/bin/activate
- on Windows
    - .\venv\Scripts\activate
- cd reinforcement/experiment-name

Follow these steps to start the training master:
- on Windows
    - Add the project directory to the PYTHONPATH enviroment variable
    - python training_master -ws
- on Unix
    - PYTHONPATH ../../ python training_master -ws

Follow these steps to start the one playing slave:
- on Windows
    - Add the project directory to the PYTHONPATH enviroment variable
    - python playing_slave -mi HOST-OF-MASTER
- on Unix
    - PYTHONPATH ../../ python playing_slave -mi HOST-OF-MASTER

Follow these steps to start tensorboard:
- cd test
- tensorboard --logdir tensorboard-logs


### Command Line Interface

You can run the command line interface to freely configure a run from the root project directory.
Mostly this will not be used directly, but by the wrappers for single experiments shown above.
The documentation is still interesting, as you can use all configurations in the wrappers for the experiments.

```
PYTHONPATH=./ python reversialphazero/command_line_interface.py -h
usage: command_line_interface.py [-h] [-nn NN_CLASS_NAME]
                                 [-d TRAINING_WORK_DIRECTORY]
                                 [-ai [AI_CLIENT]] [-w AI_CLIENT_WEIGHTS]
                                 [-i MATCH_HOST] [-p MATCH_PORT]
                                 [-m [TRAINING_MASTER]] [-s [SELFPLAY_SLAVE]]
                                 [-mi MASTER_HOST] [-mp MASTER_PORT]
                                 [-tm TRAINING_MAPS_DIRECTORY]
                                 [-ws [WEB_SERVER]]

AI Client for Reversi. Allows playing games and training Clients.

optional arguments:
  -h, --help            show this help message and exit
  -nn NN_CLASS_NAME, --nn-class-name NN_CLASS_NAME
                        The Neural Network Subclass (full name) to use for
                        this run
  -d TRAINING_WORK_DIRECTORY, --training-work-directory TRAINING_WORK_DIRECTORY
                        The directory to keep training progress
  -ai [AI_CLIENT], --ai-client [AI_CLIENT]
                        Set to true to run in AI client mode (to play a match)
  -w AI_CLIENT_WEIGHTS, --ai-client-weights AI_CLIENT_WEIGHTS
                        NN weights file to use for executing the AI client
  -i MATCH_HOST, --match-host MATCH_HOST
                        The hostname of the match/tournament server when
                        executing the AI client
  -p MATCH_PORT, --match-port MATCH_PORT
                        The port of the match/tournament server when executing
                        the AI client
  -m [TRAINING_MASTER], --training-master [TRAINING_MASTER]
                        Set to true to run as training master (performs
                        training, depends on selfplay slaves)
  -s [SELFPLAY_SLAVE], --selfplay-slave [SELFPLAY_SLAVE]
                        Set to true to run as selfplay slave (runs selfplay
                        games, reporst to a training master)
  -mi MASTER_HOST, --master-host MASTER_HOST
                        The hostname of the training master server
  -mp MASTER_PORT, --master-port MASTER_PORT
                        The port the training master runs on
  -tm TRAINING_MAPS_DIRECTORY, --training-maps-directory TRAINING_MAPS_DIRECTORY
                        The directory with all maps to be used for the
                        training run
  -ws [WEB_SERVER], --web-server [WEB_SERVER]
                        Set to true to run a monitoring webinterface on port
                        5300




