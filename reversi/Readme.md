# ReversiXT

This package contains code needed to play ReversiXT games.
This includes a representation of game states, executing moves,
networking code and some other helpers.

Please see the documentation and examples (main code at bottom of the file)
for details on how to use the package.

All modules might change in the future as they are only a very first, working implementation.
Please note that the code is not optimized at all, it's main goal is to be relatively simple
to work with without a lot of side effects.

## game_core

The `game_core` module contains all code needed to parse a map and simulate a ReversiXT game.
The `GameState` class represents a distinct state of a ReversiXT game. It holds all information
needed to identify different game states. It has methods to find all possible moves and to execute them.

To use it first create a 'Board' from a map string, then create a `GameState` from the map.


## network_core

The `network_core` module contains a `BasicServer` and `BasicClient` class.
These should be used to build any more sophisticated server or client.

Please see their respective docstrings for usage information.

## server and client

Both modules contain an example of how to use the `BasicServer` and `BasicClient`
to build an actual functional server/client.

The main code in the modules shows how to use them in example use cases.

## serialization and tournament

These modules show first approaches on how to use the code to generate data and
to player reversi tournaments to evaluate progress.
The modules are a useful start to write an automated pipeline for data generation and evaluation.