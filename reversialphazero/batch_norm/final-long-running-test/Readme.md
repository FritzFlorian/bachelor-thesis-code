# Notes

The test ran for about 175 hours on 3 PCs for Selfplay + 1 PC with Graphics Card for training and selfplay.
A tournament with 1 second turn time was played against the trivial ai.
70 vs. 60 on known maps (average 0.077).
18 vs. 72 on unknown maps (average -0.6).
These tournament results are a lot worse then the last ones. Especially the unknown map results are very bad.
This suggests that the relax on the regularization was not good. Overall the inversion rule could have added
a lot complexity. One would have to run more test with all changes tested individually to clearly see what they
change.
Another good test would be to simply try out more simulations per selfplay game.
