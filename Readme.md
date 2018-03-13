# Bachelor Thesis - Python Code

This repository is a collection of the python code used during the bachelor
thesis of Florian Fritz. The goal is to train an ai for parts of the game
ReversiXT using the AlphaZero algorithm.


![sample test run]{reversialphazero/more_maps/final-long-running-test/avg_score.png}


The repository has two main modules:
-reversi: holds all ReversiXT game logic
-reversialphazero: holds training experiments and scripts to analyze them


The main dependency, https://github.com/FritzFlorian/hometrainer, was also developed as part of
the thesis. It holds the code for the AlphaZero algorithm.


The following sections will cover installing the project and running it.
More details of both the [reversi module](reversi/Readme.md)and
the [reversialphazero module](reversialphazero/Redame.md) can
be found in the readme in their subfolders.


Please follow the [Quick Start]{#quick-start} if you know python, pip and virtualenv.
More details on specific installation steps can be found in the [troubleshooting]{#troubleshooting} section.

## Quick Start

Follow these steps to setup the repository:
- clone the repository
- create a virtualenv with `python 3.5` (or higher, 64bit)
- install the following dependencies with pip:
    - `pip install Cython`
    - `pip install matplotlib`
    - `pip install numpy`
    - `pip install pyzmq`
    - `pip install tensorflow`
    - `pip install hometrainer`
- optionally uninstall tensorflow and replace it with `tensorflow-gpu`


After the initial setup you will have to generate certificates for a secure network communication:
- `python keys/generate_certificates.py`
- Distribute the generated keys in the directory `keys` to all machines that should participate in the training
- For more Information, see [the Readme in the keys directory]{keys/Readme.md}


To run an experiment do the following:
- `cd project/reversialhpazero/experiment_name`
- `PYTHONPATH=../../ python training_master.py`
- `PYTHONPATH=../../ python playing_slave.py`

Results of this run will be placed in an folder relative to the shell session called `test`.


## Troubleshooting

This section contains further links and help to setup everything and resolve typical problems.
It is structured by topics, most of the time there will be some special instructions for windows.

### Virtualenv

To manage dependencies it is best to create an virtual environment for each project.
This is a self contained python environment only used locally by one project.
The following will contain the main steps to use virtualenv, for details visit https://virtualenv.pypa.io/en/stable/.


Creating a new virtualenv:
- `cd project`
- `pip install virtualenv`
- `virtualenv -p python3 venv`
    - `python3` must point to the correct python 3.5 or higher installation
    - `python3` can also be a absolute path tho python 3.5 or higher

Using the virtualenv:
- Before performing any task in the project you should activate the virtualenv
- On Unix/Mac:
    - `cd project`
    - `source venv/bin/activate`
- On Windows:
    - Do this ONCE in an admin power shell `Set-ExecutionPolicy RemoteSigned`
    - `cd project`
    - `.\venv\Scripts\activate`


The virtualenv will manage all dependencies locally, so be sure to activate it before performing
any tasks with the project.
You can use alternatives like anaconda on windows if you want to. Please see their instructions if
you want to do that.


### Installing Dependencies

The installation of the dependencies is usually done using the pip command line tool.
[Quick Start]{#quick-start} lists the required dependencies.
This should work fine on Unix/Mac, on Windows you might need to do the following:
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


### Running Scripts

To run scripts and the training you will need to add the project directory to the PYTHONPATH
environment variable. On Unix/Mac you ca usually do this 'inline' with your commands
by using something similar to `PYTHONPATH=..\..\ python script.py`.

On Windows the simplest way is to just add the project directory to your PYTHONPATH system variable.


### Tensorflow with GPU Support

If running the training master with GPU acceleration follow these instructions:
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


### Updating the project/codebase

To update the code you will usually pull the git repo and update the hometrainer dependency.
The following commands will do this:
- `cd project`
- `source venv/bin/activate`
- `git pull`
- `pip install hometrainer -U`


### Run the Project
The main experiments that are run during the bachelor thesis can be found in the subdirectories
`reversialphazero/experiment_name`. To start training you will need to start one 'training_master' process
(best run on a machine with fast GPU) and at least one 'playing_slave'.
Most variables to run these are pre-configured in the separate experiments, to run them you normally only have
to provide the ip of the 'training_master'.

Before performing any of the below task follow these steps:
- `cd project`
- on Unix
    - `source venv/bin/activate`
- on Windows
    - `.\venv\Scripts\activate`
- `cd reversialphazero/experiment-name`

Follow these steps to start the training master:
- on Windows
    - `python training_master -ws`
- on Unix
    - `PYTHONPATH=../../ python training_master -ws`

Follow these steps to start the one playing slave:
- on Windows
    - `python playing_slave -mi HOST-OF-MASTER`
- on Unix
    - `PYTHONPATH=../../ python playing_slave -mi HOST-OF-MASTER`

Follow these steps to start tensorboard:
    - `tensorboard --logdir test/tensorboard-logs`


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




