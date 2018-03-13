# Simple 8 by 8

This experiment was used to understand the algorithm and to get all basic parts that are needed running.
The project structure changed a lot since this test, to run it you need to revert the repository
or recreate the settings of this test with the new project structure.

## General Structure

This first experiment executes the different phases of the training process serialized in different epochs.
Each epoch the following steps are executed:
- A set amount of selfplay games are played using the current best neural network.
  The generated selfplay data is stored on disk.
- The results of the selfplay games are used to train the current best network to a new network for
  this epoch. This network is saved on disk.
- The new network plays a small tournament against the current best network. If it wins by a set margin
  it is used as the new best network.
- If the new network was better then the last one a small tournament is played against the trivial AI.
  This is used to see the progress against an external client to see if the network actually plays good.

## Improving the code

One of the most important things to take from this simple experiment where improvements in code.
Bugs where fixed and performance bottlenecks identified.
After this simple experiment the code is proven to work and most parameters have sane values to
continue with further experiments.

## Improved understanding of the problem and algorithm

The test also helped to fully understand the algorithm used for training.
One example are random rotations of the game field that where added as a consequence of
overfitting to the training examples.

In general the work with this serialized variant of the code acts as a basis to
adopt it to a more complicated version of ReversiXT and to further improve the training pipeline.

## Next Steps

### Use more ReversiXT rules

The next step will be to allow more ReversiXT rules, like bonus stones or custom maps.
This is now simpler to achieve as the base concept works.

### Distributed execution/continuous training

The tests with the simple 8 by 8 board already showed that a lot runtime is needed to properly train a model.
One interesting fact is, that the training itself is not the only time consuming factor.
The selfplay, conversion of the input for training and the evaluation are currently more time consuming
then the training of the model. To fix this it would make a lot sense to allow the selfplay to be executed
on a separate system in parallel. This could reduce the training time by a factor of 2 or 3 and would therefore
help to experiment with more parameter configurations.

A second important change is to stop using fixed 'iterations'. The alphago zero paper suggests full parallel selfplay
that is not separated into interations. This way the training can always be executed on the last X training sets.
One further benefit of this is that training for 'failed' iterations is not waisted, but continued until a
good point is reached.

## Further Findings

Random rotations greatly improve both, exploration during selfplay and generalization during training.

Residual blocks greatly improve training speed with deep networks. Deep networks where not even able to
train at all without the skip connections from the residual blocks.

## Final Time Measurements

This is a note of execution times using the final Neural Network Structure/Variables used in the final run.
It should help me to see where I should work on next.

- Selfplay:     10 minutes          35 games, 128 simulations per move
- Training:     10 minutes          2500 batches, 128 evaluations per batch
- Evaluation:   4  minutes          21 games, 48 simulations per move
- AI-Games:     4  minutes          21 games, 1 second per move
