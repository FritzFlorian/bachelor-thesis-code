# ReversiXT

This package contains code needed to play ReversiXT games.
This includes a representation of game states, executing moves,
networking code and some other helpers.


Please see the documentation and examples (main code at bottom of the file)
for details on how to use the package.


To make sure that the moves work there are very detailed unit tests in the `test`
directory. If you make any changes, be sure to run the tests. They cover most
edge cases and should be a solid baseline for further development.
Please add more tests if you find new edge cases.


The code uses cython for some speedups. Please note that the code is 'grown',
maybe more performance and readability could be reached by re-writing it with the
knowledge from the first implementation.



## game_core

The `game_core` module contains all code needed to parse a map and simulate a ReversiXT game.
The `GameState` class represents a distinct state of a ReversiXT game. It holds all information
needed to identify different game states. It has methods to find all possible moves and to execute them.

To use it first create a 'Board' from a map string, then create a `GameState` from the map.

## network_core

The `network_core` module contains a `BasicServer` and `BasicClient` class.
These should be used to build any more sophisticated server or client.

Please see their respective docstrings for usage information.
