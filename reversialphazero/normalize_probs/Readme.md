# Normalize Probs

Next experiment after 'more_maps'.
The following changes where applied.

- Use more maps and add inversion rule (13 maps, 5 with inversion)
- Clean out the output of the neural network to only add valid moves to the probability distribution.
- Change c_puct form 3 to 1 to fit the 'cleaned' move probabilities
- regularization was lowered (both L2 and dropout)

## Testrun

The test ran for about 175 hours on 3 PCs for Selfplay + 1 PC with Graphics Card for training and selfplay.


## Results

A tournament with 1 second turn time was played against the trivial ai.


Results:
- 70 vs. 60 on known maps (average 0.077).
- 18 vs. 72 on unknown maps (average -0.6).

These tournament results are a lot worse then the last ones. Especially the unknown map results are very bad.
This suggests that the relax on the regularization was not good. Overall the inversion rule could have added
a lot complexity. One would have to run more test with all changes tested individually to clearly see what they
change.
Another good test would be to simply try out more simulations per selfplay game.


Overall more analyzing tools are needed to further look into why the playing strength is not increasing.

