Next experiment after 'distributed_8_by_8' that ran with a single map.
The following changes where applied.

- Use multiple maps
    We will play on a total of 5 different maps. This will show how the AI performs on learning more general
    rules than that corners are good on the simple 8 by 8 map.
    It might also help with over-fitting to one specific tactic, which seemed to be the problem in the simple 8
    by 8 example (see graphs with marked regions)
- Increase L2 Loss a little
    This is also to prevent overfitting, as the L2 loss 'spiked' when the results of the last run got
    worse again (see marked regions in the graphs).
- Add 'deep network' skip connections with raw input to every residual block
    This could help to reduce some wasted convolutional blocks for 'passthrough' of simple information,
    like the legal moves.
- Add two more residual blocks
    Maybe the worse results/tactic came because the nn could not sufficiently learn general game rules.
    Also the board got bigger in the 10 by 10 map.
