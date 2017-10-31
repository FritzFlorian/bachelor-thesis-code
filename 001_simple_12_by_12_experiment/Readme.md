# 12x12 Simple Map - Count Stones after one Move

This is my initial test to train a model related to reversi.
The goal is to train the model to predict correct moves for player one and how many stones
player one will own after the move.

## Input

The input is a 12x12 standard reversi board with two players.
Input will be one hot encoded and differenciate between:
- Player One
- Player Two
- No Player

The input will be fed as a 2D data structure with 3 layers.

## Output

The desired output is a 12x12 array. Each field should contain the
number of stones player one ownes after it executed a single move on
the given field.
Fields that are no valid moves output -10.

## Goal

Get confortable with tensorflow and get a first test on how well a model can
be trained to understand the basic rules of reversi.

## Subfolders/Subprojects

Short explanations of the different subfolders/subprojects.

### data_generation

Used to execute selfplay between two ai_trival clients on a 12x12 standard map.
The games are played at search depth 1 to 10, logs are saved to a logs folder.

This data is then processed using a reversi java client and converted to csv
files containing the inptu/output needen for training the model. The client
creates an `x_values.csv` containing the map (0 = no player, 1 = player one,
2 = player two) and an `y_values.csv` containing the target values.
The target values are 0 if the move is not possible and the number of stones
(normalized between 0 and 5) after player one moved on the given field.

### model_training

## Conclusion

This 'mini project' helped me to learn the basics of tensorflow and
to get confortable with working with python.

The results of the final round of trainig are promising, but also show
some weakneses of this first round of training:
- The model successfully learend to mark valid moves with a noticably higher value
then other fields
- The model generalized well to the test set
- The generalization is only a partly success: it seems that because all samples are from
the same or very simmilar games it was easier for the model to generalize
- The model only generalized after I bumped the training set size from about 1k to about 22k
samples
- The model seems to not care about how many stones the player ownes after the move.
It seems like it's going for avarage values in the sample data rathen then actually counting.

All in all the resuts are rather promising. Without much more then some 'standard' convolutional
layers (as suggessted by other work on games and image processing) are enough to train the
model to understand the basic concept of standard reversi, wich also suggests that this can
be specifified to train it to predict good moves rather than only possible moves.
