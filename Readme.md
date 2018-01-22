# Bachelor Thesis - Python Code

This repository is a collection of the python code used during the bachelor
thesis of florian fritz.

Each subfolder can be seen as an individual module or experiment.

The Readme documents in each subfolder are used as documentation for myself to be able
to write/reason about what I did implement/test for the thesis.

This Readme is a highlevel view on all experiments and tasks I did and it's main
purpose is to make sure that I keep track of all tasks done in case I later intend to
use findings from these tasks in my thesis.

## Tasks/Experiments

### 001_simple_12_by_12_experiment

Goal was to train a model to predict stone counts after player one moved on a simple 
12 by 12 standard reversi board.
I was able to train a model that could predict possible moves very well.

It showed that the size of the test set is very important to generalize.

### Compile tensorflow for faster processing

Compiled tensorflow from source because this is supposed to speed up some  operations.
Improvement was only about 10%. It seems like the speed up is manly in special operations
for image processing/detection problems.

### Reversi - Game Core, Network Core, Tournament

When working with the 001_simple_12_by_12_experiment it was very clear that I need an
python implementation of reversi for both data gathering and the actual AI client later on.
This is important as it it impractical to mix my legacy Java Client (from the ReversiXT course)
with python code for the actual ML tasks.

The reversi module contains all code needed to work with ReversiXT game states, interact with
other clients/servers via the network protocol specific to the ReversiXT lecture, to generate training
data and to evaluate training results in tournaments.

This module will hopefully make future work on the project easier.


### Expert Iteration/Pure Monte Carol Tree Search Reinforcement

The simple_8_by_8 test should show how well the expert iteration presented in deepminds
AlphaGoZero paper works on a simple 8 by 8 reversi board.

The results where quite pleasing (see the final run stats.csv). The AI managed to learn
to get about a 70 percent winrate against the trivial AI by pure selfplay.
This was learning in about 6 hours of training on an laptop, so it was also fairly fast.

The main point of this experiment was to get myself comfortable with the general
algorithm and to make sure my implementation is bug free.


### Speed up and restructure

The simple 8 by 8 experiment showed that the slowest part is not the neural network training or execution,
but the python code executing the moves. This is not very good, as under this circumstances adding a
GPU would not significantly speed up the training progress, making it unfeasible slow for bigger test runs.

This led to some research on how to speed up my python code.
The first idea was to execute the non tensorflow code using pypy and pass neural network execution
to an cpython process. This is necessary as tensorflow does not work with pypy.
To make this separation quite some refactoring was required.
Sadly pypy did not bring the desired speedup, as it is extremly slow when manually working with
numpy arrays.

Because of that Cython was tested as a further alternative to speed up the code.
This was a success and gained about 50 percent in speed (after adding cython types to game core).

The new code structure will be kept anyway, as it actually makes working with neural networks easier
in the way I use them in this project.


### Distributed 8 by 8

The code was changed to allow one machine with a GPU to do the training and multiple other machines to do
the selfplay games. This helps a lot, as with the current network size the selfplay is mainly the bottleneck,
not the training (when a gpu is used).

The change allowed to more then double the speed at which experiments can be run. The restructuring also
helped a lot to make the code easier to maintain and therefore to more quickly test out new parameters/networks.

After some tweaking and bug fixing the distributed 8 by 8 ran very smoothly on my laptop combined with my desktop pc.

The results where also promising, showing a very strong play after about 15 hours of training.
(see the final run for graphs/details)


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




