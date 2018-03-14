# More Maps

Next experiment after 'multiple_maps'.
The following changes where applied.


Major changes:
- Use more maps
    More maps seemed to help to generalize in the last run, so increasing them should not hurt to help the AI
    find a good strategy.
- Double the amount of filters per CNN layer
    The last run seemed to stop learning more at some point, so increasing the filter number could
    help it learn more features.
- Increase Number of layers given to final fully connected layer
- Batch size was increased from 64 to 200. Look at last 200 selfplay games instead of 128

Minor changes (did not really have any effect):
- Adjust the L2 weight to the new, more complex network
    The number of weights in the network nearly doubled, so the L2 loss has to be ruffly halved to
    stay at the same relevance as before.
- Add the input-skip connections to the output
    This might further help to not 'waste' some of the connections to the fully connected layer
    with figuring out correct moves. (seems to make no big difference)
- Try to use the adam optimizer instead of momentum
    This should help to adapt the learning rate. I'm not sure if this will work well, as we do not
    have a classical supervised learning task, but it's worth a shot as it usually performs better
    then the momentum optimizer. -> It did not work at all (spikes in loss), so Momentum was used in the final run

## Testrun

The run took about 145 hours using three older PC's to play the games and one
modern PC with a 1080 TI for training and also game execution.
The test ran about 175 hours.


## Results

The results do not really differ from the last run,
this suggests that the stagnating training progress was not due to the NN but because
of settings for the Alpha Zero Tree Search.

