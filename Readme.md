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

- Install Python 3.5 64 bit
- Install virtualenv (via pip)
- Clone the project
- Create a virtualenv for the project (cd project-folder; virtualenv -p python3 venv; source .\venv\bin\activate)
- If running on windows
    - Install required C++ compiler for cython
    - This should work if you use the exact versions stated above (make sure to check Windows 8.1 SDK):
      http://landinghub.visualstudio.com/visual-cpp-build-tools
    - More Details: https://github.com/cython/cython/wiki/CythonExtensionsOnWindows
- Run `pip install -r requirements.txt`
- If running the training master with GPU acceleration
    - Install all needed CUDA drivers (as stated on the TF homepage, make sure to get the right ones)
    - Do not forget to set your PATH (also stated on the TF homepage)
    - `pip uninstall tensorflow`
    - `pip install tensorflow-gpu`
