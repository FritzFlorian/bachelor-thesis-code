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

