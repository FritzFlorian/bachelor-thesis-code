# Changes

Next experiment after 'multiple_maps'.
The following changes where applied.

- Use more maps
    More maps seemed to help to generalize in the last run, so increasing them should not hurt to help the AI
    find a good strategy.
- Adjust the L2 weight to the new, more complex network
    The number of weights in the network nearly doubled, so the L2 loss has to be ruffly halved to
    stay at the same relevance as before.
- Double the amount of filters per CNN layer
    The last run seemed to stop learning more at some point, so increasing the filter number could
    help it learn more features.
- Add the input-skip connections to the output
    This might further help to not 'waste' some of the connections to the fully connected layer
    with figuring out correct moves.
- Increase Number of layers given to final fully connected layer


