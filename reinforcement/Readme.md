# Pure Reinforcement Training

This module implements the basic technique used in alpho go zero.
It trains itself by using a reinforcement learning algorithm based on monte
carlo tree search.

It does not do any supervised learning, it is trained solely by selfplay.

## Notes

### Game Scores

Games can not be scored regarding standard reversi tournament rules.
In a standard rules players are awarded the score of the higher rank if a tie occurred.

Example:
In a two player game usually the winner gets 25 points and the looser gets 11 points.
If both have the same stone count both get 25 points.

This is a problem, as it destroys the zero-sum-game property.
The AI could potentially learn in selfplay that it is the best strategy to always play for
an perfect tie.
This would work in the training phase, as it's only enemy is itself.

Later on when playing against an external client this strategy will fall apart and
the trained AI is quite useless.
